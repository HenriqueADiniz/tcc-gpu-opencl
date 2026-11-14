/*
 * teto_vram.cpp — Os ~222 GB/s do Exp. 2 sao o teto da VRAM? Valem para outros
 * tamanhos de buffer? O resultado de 1 MB depende de o L2 ja conter o buffer?
 *
 * O Exp. 2 (slc_l2.cpp) mediu dois tamanhos: 1 MB (cabe na L2 de 2 MB) e 64 MB
 * (nao cabe). Este programa estende a medida em tres direcoes:
 *
 *   A) Varredura do working set, de 256 KB a 2 GB, com os MESMOS tres kernels
 *      de slc_l2.cl (arquivo intocado), mesmo GS, mesmo work-group, mesma ordem
 *      (sem flag, glc, glc slc), 25 execucoes e 3 de aquecimento. So muda o
 *      tamanho -- e R acima de 64 MB, para manter duas passadas sobre o
 *      buffer, como no caso de 64 MB.
 *
 *   B) Teto pratico: o kernel do Exp. 2 tem 256 B em voo por wavefront. Se os
 *      ~222 GB/s fossem limite do kernel, e nao da memoria, kernels com 4x a
 *      16x mais bytes em voo (teto_vram.cl) e mais work-items passariam disso.
 *      Medido em 1 GB (VRAM) e em 1 MB (L2; so variantes glc slc, que nao
 *      passam pelo L1).
 *
 *   C) Em slc_l2.cpp o kernel glc slc roda depois de dois kernels que ja
 *      deixaram o buffer no L2. Aqui, antes de CADA execucao medida, outro
 *      kernel le um buffer de 64 MB (duas passadas), que expulsa o anterior
 *      do L2; o tempo medido e so o do kernel de interesse.
 *
 *   D) O caso de 1 MB repetido em 10 buffers distintos, para saber quanto o
 *      valor da Tabela 4 varia com a alocacao.
 *
 * Pico teorico da GDDR5 desta placa: 2000 MHz (pp_dpm_mclk) x 4 transferencias
 * x 256 bits / 8 = 256 GB/s. A saida de todo kernel e conferida contra o valor
 * esperado (buffer de 1.0f). Antes de medir cada buffer, ~0,2 s de leitura
 * dele leva os clocks ao regime de carga (o preenchimento de um buffer grande
 * deixa a GPU ociosa por segundos).
 *
 * Compilar e rodar (a partir da raiz do projeto):
 *   source setup/env_rocm.sh
 *   g++ -Wall -Wextra -O2 experimentos/teto_vram.cpp -o output/teto_vram -lOpenCL
 *   ./output/teto_vram [saida.csv] | tee resultados/teto_vram.txt
 * O CSV (padrao resultados/teto_vram.csv) nunca e sobrescrito: se existir, o
 * programa para. O binario do programa vai para isa/rocm_gfx803_teto_vram.bin.
 */
#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <CL/cl.h>

#define CK(x) do{cl_int _e=(x);if(_e!=CL_SUCCESS){fprintf(stderr,"%s:%d: erro OpenCL %d\n",__FILE__,__LINE__,_e);exit(1);}}while(0)
static char* rd(const char*p){FILE*f=fopen(p,"rb");if(!f){fprintf(stderr,"nao abriu %s\n",p);exit(1);}fseek(f,0,SEEK_END);long n=ftell(f);fseek(f,0,SEEK_SET);char*b=(char*)malloc(n+1);size_t k=fread(b,1,n,f);b[k]=0;fclose(f);return b;}
static int cmpd(const void*a,const void*b){double x=*(const double*)a,y=*(const double*)b;return (x>y)-(x<y);}
static double agora(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}

static const double PICO = 256.0;                  // GB/s teoricos da GDDR5
static const int RUNS = 25, AQUEC = 3, LS = 256;   // como em slc_l2.cpp
static const int GS = 65536, GSMAX = 262144;
static cl_context ctx; static cl_command_queue q; static cl_program pr; static FILE* csv;
static cl_mem dO;   // saida, GSMAX floats

// Buffer de W bytes com 1.0f, criado como em slc_l2.cpp (COPY_HOST_PTR, na VRAM).
static cl_mem buffer(size_t W){
    float*h=(float*)malloc(W); if(!h){fprintf(stderr,"sem RAM para %zu B\n",W);exit(1);}
    for(size_t i=0;i<W/4;i++)h[i]=1.0f;
    cl_int e; cl_mem d=clCreateBuffer(ctx,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,W,h,&e); CK(e);
    free(h); return d;
}

static cl_kernel kernel(const char*k,cl_mem dA,cl_mem out,int mask,int R,int gs){
    cl_int e; cl_kernel kn=clCreateKernel(pr,k,&e); CK(e);
    CK(clSetKernelArg(kn,0,sizeof(cl_mem),&dA)); CK(clSetKernelArg(kn,1,sizeof(cl_mem),&out));
    CK(clSetKernelArg(kn,2,sizeof(int),&mask)); CK(clSetKernelArg(kn,3,sizeof(int),&R));
    CK(clSetKernelArg(kn,4,sizeof(int),&gs));
    return kn;
}

// ~s segundos de sweep_noflag sobre o buffer, para os clocks entrarem em regime.
static void aquece(cl_mem dA,size_t W,double s){
    cl_kernel kn=kernel("sweep_noflag",dA,dO,(int)(W/4-1),512,GS); size_t g=GS,l=LS;
    for(double t0=agora();agora()-t0<s;){
        for(int r=0;r<8;r++) CK(clEnqueueNDRangeKernel(q,kn,1,NULL,&g,&l,0,NULL,NULL));
        CK(clFinish(q));
    }
    clReleaseKernel(kn);
}

static const char* rotulo(size_t W){
    static char s[32];
    if(W<((size_t)1<<20)) snprintf(s,32,"%zu KB",W>>10);
    else if(W<((size_t)1<<30)) snprintf(s,32,"%zu MB",W>>20);
    else snprintf(s,32,"%zu GB",W>>30);
    return s;
}

// Executa AQUEC+RUNS vezes um kernel de assinatura (A, out, mask, R, gsize) com
// u cargas de vb bytes por iteracao (voo: bytes em voo por wavefront, da ISA),
// confere a saida e grava uma linha no CSV. Se fl != NULL, fl roda e TERMINA
// antes de cada execucao: enfileirado logo atras de outro kernel, o carimbo
// START do ROCm sai antes do fim do anterior e o tempo medido o incluiria.
// Devolve a vazao mediana em GB/s; em *gbmax, a da execucao mais rapida.
static double mede(const char* parte, int buf, const char* k, const char* flags, int vb, int u, int voo,
                   int gs, size_t W, int R, cl_mem dA, cl_kernel fl, double* gbmax){
    cl_kernel kn=kernel(k,dA,dO,(int)(W/vb-1),R,gs);
    size_t g=gs,l=LS,gf=GS; double t[RUNS];
    for(int r=0;r<RUNS+AQUEC;r++){
        if(fl){ CK(clEnqueueNDRangeKernel(q,fl,1,NULL,&gf,&l,0,NULL,NULL)); CK(clFinish(q)); }
        cl_event ev; CK(clEnqueueNDRangeKernel(q,kn,1,NULL,&g,&l,0,NULL,&ev));
        CK(clWaitForEvents(1,&ev)); cl_ulong a,b;
        CK(clGetEventProfilingInfo(ev,CL_PROFILING_COMMAND_START,sizeof(a),&a,NULL));
        CK(clGetEventProfilingInfo(ev,CL_PROFILING_COMMAND_END,sizeof(b),&b,NULL));
        if(r>=AQUEC)t[r-AQUEC]=(b-a)/1e6;
        clReleaseEvent(ev);
    }
    // cada carga traz vb/4 floats iguais a 1.0
    float*h=(float*)malloc((size_t)gs*4);
    CK(clEnqueueReadBuffer(q,dO,CL_TRUE,0,(size_t)gs*4,h,0,NULL,NULL));
    float esp=(float)R*u*(vb/4); int erros=0;
    for(int i=0;i<gs;i++) if(h[i]!=esp) erros++;
    free(h);
    if(erros){fprintf(stderr,"%s: %d saidas erradas (esperado %.0f)\n",k,erros,esp);exit(1);}
    qsort(t,RUNS,sizeof(double),cmpd);
    double bytes=(double)gs*R*u*vb, gb=bytes/(t[RUNS/2]/1e3)/1e9;
    *gbmax=bytes/(t[0]/1e3)/1e9;
    fprintf(csv,"%s,%d,%s,%s,%d,%d,%d,%zu,%d,%.0f,%.4f,%.4f,%.4f,%.1f,%.1f,%.3f\n",
            parte,buf,k,flags,vb,voo,gs,W,R,bytes,t[RUNS/2],t[0],t[RUNS-1],gb,*gbmax,gb/PICO);
    fflush(csv);
    clReleaseKernel(kn);
    return gb;
}

static const char*ks[]={"sweep_noflag","sweep_glc","sweep_glc_slc"};
static const char*fl[]={"GLC=0 SLC=0","GLC=1 SLC=0","GLC=1 SLC=1"};

// Os tres kernels de slc_l2.cl, na ordem de slc_l2.cpp, sobre o buffer dA de W bytes.
static void tres(const char* parte,int buf,const char* rot,size_t W,cl_mem dA,cl_kernel flush){
    size_t r2=2*W/((size_t)GS*4); int R=(int)(r2>512?r2:512);
    aquece(dA,W,0.2); double gb[3],mx;
    for(int k=0;k<3;k++) gb[k]=mede(parte,buf,ks[k],fl[k],4,1,256,GS,W,R,dA,flush,&mx);
    printf("%9s %6d %12.1f %12.1f %12.1f %8.3fx\n",rot,R,gb[0],gb[1],gb[2],gb[1]/gb[2]);
}

int main(int argc,char**argv){
    const char* SAIDA = argc>1?argv[1]:"resultados/teto_vram.csv";
    csv=fopen(SAIDA,"wx");
    if(!csv){fprintf(stderr,"%s ja existe ou nao abre: CSV de campanha nao e sobrescrito\n",SAIDA);return 1;}
    fprintf(csv,"parte,buf,kernel,flags,bytes_por_load,bytes_em_voo_por_wave,gs,ws_bytes,R,"
                "bytes_lidos,mediana_ms,min_ms,max_ms,gbs_mediana,gbs_max,frac_pico\n");

    cl_uint np; CK(clGetPlatformIDs(0,NULL,&np));
    cl_platform_id*pl=(cl_platform_id*)malloc(np*sizeof(*pl)); CK(clGetPlatformIDs(np,pl,NULL));
    cl_device_id dev=NULL; for(cl_uint i=0;i<np;i++)if(clGetDeviceIDs(pl[i],CL_DEVICE_TYPE_GPU,1,&dev,NULL)==CL_SUCCESS)break;
    cl_int e; ctx=clCreateContext(NULL,1,&dev,NULL,NULL,&e); CK(e);
    q=clCreateCommandQueue(ctx,dev,CL_QUEUE_PROFILING_ENABLE,&e); CK(e);
    char nome[256]; CK(clGetDeviceInfo(dev,CL_DEVICE_NAME,sizeof(nome),nome,NULL));
    printf("Dispositivo: %s; pico teorico da GDDR5: %.0f GB/s\n",nome,PICO);

    const char*src[]={rd("experimentos/slc_l2.cl"),rd("experimentos/teto_vram.cl")};
    pr=clCreateProgramWithSource(ctx,2,src,NULL,&e); CK(e);
    if(clBuildProgram(pr,1,&dev,"",NULL,NULL)!=CL_SUCCESS){size_t ls;clGetProgramBuildInfo(pr,dev,CL_PROGRAM_BUILD_LOG,0,NULL,&ls);char*lg=(char*)malloc(ls+1);clGetProgramBuildInfo(pr,dev,CL_PROGRAM_BUILD_LOG,ls,lg,NULL);lg[ls]=0;fprintf(stderr,"%s\n",lg);return 1;}
    size_t bs; CK(clGetProgramInfo(pr,CL_PROGRAM_BINARY_SIZES,sizeof(bs),&bs,NULL));
    unsigned char*bin=(unsigned char*)malloc(bs); CK(clGetProgramInfo(pr,CL_PROGRAM_BINARIES,sizeof(bin),&bin,NULL));
    FILE*fb=fopen("isa/rocm_gfx803_teto_vram.bin","wb"); if(fb){fwrite(bin,1,bs,fb);fclose(fb);} free(bin);
    dO=clCreateBuffer(ctx,CL_MEM_WRITE_ONLY,(size_t)GSMAX*4,NULL,&e); CK(e);

    // A) mesmos kernels de slc_l2.cl, so o tamanho do buffer muda
    printf("\nA) Varredura do working set (kernels de slc_l2.cl, GS=%d, mediana de %d execucoes, GB/s)\n",GS,RUNS);
    printf("%9s %6s %12s %12s %12s %9s\n","buffer","R","GLC=0 SLC=0","glc","glc slc","slc/glc");
    for(size_t W=(size_t)256<<10; W<=(size_t)2<<30; W*=2){
        cl_mem dA=buffer(W); tres("A",0,rotulo(W),W,dA,NULL); clReleaseMemObject(dA);
    }

    // B) mais bytes em voo por wavefront; R da ~2 passadas em 1 GB e 128 MiB em 1 MB
    struct { const char* k; const char* fl; int vb,u,voo,gs; size_t W; } cb[] = {
        {"sweep_noflag",     "GLC=0 SLC=0", 4,1, 256, 65536,(size_t)1<<30},
        {"le_f1_u4",         "GLC=0 SLC=0", 4,4,1024, 65536,(size_t)1<<30},
        {"le_f4_u1",         "GLC=0 SLC=0",16,1,1024, 65536,(size_t)1<<30},
        {"le_f4_u4",         "GLC=0 SLC=0",16,4,2048, 16384,(size_t)1<<30},
        {"le_f4_u4",         "GLC=0 SLC=0",16,4,2048, 65536,(size_t)1<<30},
        {"le_f4_u4",         "GLC=0 SLC=0",16,4,2048,262144,(size_t)1<<30},
        {"asm_f4_u4",        "GLC=0 SLC=0",16,4,4096, 16384,(size_t)1<<30},
        {"asm_f4_u4",        "GLC=0 SLC=0",16,4,4096, 65536,(size_t)1<<30},
        {"le_f4_u4_nt",      "GLC=1 SLC=1",16,4,2048, 65536,(size_t)1<<30},
        {"asm_f4_u4_glc_slc","GLC=1 SLC=1",16,4,4096, 65536,(size_t)1<<30},
        {"sweep_glc_slc",    "GLC=1 SLC=1", 4,1, 256, 65536,(size_t)1<<20},
        {"le_f4_u4_nt",      "GLC=1 SLC=1",16,4,2048, 16384,(size_t)1<<20},
        {"asm_f4_u4_glc_slc","GLC=1 SLC=1",16,4,4096, 16384,(size_t)1<<20},
    };
    printf("\nB) Bytes em voo por wavefront (mediana e melhor de %d execucoes)\n",RUNS);
    printf("%-6s %-18s %-12s %7s %7s %10s %10s %7s\n","buffer","kernel","flags","B/wave","GS","GB/s med","GB/s max","%pico");
    cl_mem dA=NULL; size_t Wa=0;
    for(size_t c=0;c<sizeof(cb)/sizeof(cb[0]);c++){
        if(cb[c].W!=Wa){ if(dA)clReleaseMemObject(dA); Wa=cb[c].W; dA=buffer(Wa); aquece(dA,Wa,0.2); }
        size_t alvo=2*Wa>((size_t)128<<20)?2*Wa:((size_t)128<<20);
        int R=(int)(alvo/((size_t)cb[c].gs*cb[c].u*cb[c].vb)); double mx;
        double gb=mede("B",0,cb[c].k,cb[c].fl,cb[c].vb,cb[c].u,cb[c].voo,cb[c].gs,Wa,R,dA,NULL,&mx);
        printf("%-6s %-18s %-12s %7d %7d %10.1f %10.1f %6.1f%%\n",rotulo(Wa),cb[c].k,cb[c].fl,
               cb[c].voo,cb[c].gs,gb,mx,100*gb/PICO);
    }
    clReleaseMemObject(dA);

    // C) como A, mas com o L2 ocupado por outro buffer antes de cada execucao
    const size_t WF=(size_t)64<<20; cl_mem dF=buffer(WF);
    cl_mem dFo=clCreateBuffer(ctx,CL_MEM_WRITE_ONLY,(size_t)GS*4,NULL,&e); CK(e);
    cl_kernel kf=kernel("sweep_noflag",dF,dFo,(int)(WF/4-1),512,GS);
    printf("\nC) Antes de cada execucao, sweep_noflag le outro buffer de 64 MB (fora do tempo medido)\n");
    printf("%9s %6s %12s %12s %12s %9s\n","buffer","R","GLC=0 SLC=0","glc","glc slc","slc/glc");
    for(size_t W=(size_t)256<<10; W<=(size_t)64<<20; W*=4){
        cl_mem dA=buffer(W); tres("C",0,rotulo(W),W,dA,kf); clReleaseMemObject(dA);
    }
    clReleaseKernel(kf); clReleaseMemObject(dFo); clReleaseMemObject(dF);

    // D) o caso de 1 MB da Tabela 4 em buffers distintos, todos vivos ao mesmo
    //    tempo (enderecos fisicos diferentes), com o protocolo de A
    const int ND=10; const size_t W1=(size_t)1<<20; cl_mem bd[ND];
    for(int i=0;i<ND;i++) bd[i]=buffer(W1);
    printf("\nD) 1 MB em %d buffers distintos, protocolo de A\n",ND);
    printf("%9s %6s %12s %12s %12s %9s\n","buffer","R","GLC=0 SLC=0","glc","glc slc","slc/glc");
    for(int i=0;i<ND;i++){
        char rot[16]; snprintf(rot,16,"1 MB #%d",i+1);
        tres("D",i+1,rot,W1,bd[i],NULL);
    }
    for(int i=0;i<ND;i++) clReleaseMemObject(bd[i]);

    clReleaseMemObject(dO); fclose(csv);
    printf("\nCSV: %s\n",SAIDA);
    return 0;
}
