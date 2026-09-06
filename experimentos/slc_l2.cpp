/*
 * slc_l2.cpp — SLC=1 ignora a cache L2 no GFX8? (resposta: nao)
 *
 * Varre um buffer repetidas vezes com os tres caminhos de cache e compara
 * dois regimes:
 *
 *   A) buffer de 1 MB  — cabe na L2 de 2 MB da Ellesmere
 *   B) buffer de 64 MB — nao cabe, todo acesso vai a VRAM
 *
 * O caso B calibra a bancada: mostra quanto vale, em GB/s, "vir da VRAM".
 * O caso A e o teste: se SLC=1 desligasse a L2, o tempo do kernel glc+slc
 * cairia para o patamar do caso B. Se SLC=1 for apenas dica de eviccao, os
 * tres kernels ficam colados, porque com o buffer inteiro residente nao ha
 * o que evictar.
 *
 * Os tres kernels usam inline asm, entao a codegen e a mesma e a unica
 * diferenca entre eles sao os bits de cache da instrucao.
 *
 * Compilar e rodar (a partir da raiz do projeto):
 *   source setup/env_rocm.sh
 *   g++ -Wall -O2 experimentos/slc_l2.cpp -o output/slc_l2 -lOpenCL
 *   ./output/slc_l2 experimentos/slc_l2.cl
 */
#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <CL/cl.h>
static char* rd(const char*p){FILE*f=fopen(p,"rb");fseek(f,0,SEEK_END);long n=ftell(f);fseek(f,0,SEEK_SET);char*b=(char*)malloc(n+1);size_t k=fread(b,1,n,f);b[k]=0;fclose(f);return b;}
int cmpd(const void*a,const void*b){double x=*(const double*)a,y=*(const double*)b;return (x>y)-(x<y);}
int main(int argc,char**argv){
    const char* clp = argc>1?argv[1]:"experimentos/slc_l2.cl";
    cl_uint np;clGetPlatformIDs(0,NULL,&np);
    cl_platform_id*pl=(cl_platform_id*)malloc(np*sizeof(*pl));clGetPlatformIDs(np,pl,NULL);
    cl_device_id dev=NULL;for(cl_uint i=0;i<np;i++)if(clGetDeviceIDs(pl[i],CL_DEVICE_TYPE_GPU,1,&dev,NULL)==CL_SUCCESS)break;
    cl_int e;cl_context ctx=clCreateContext(NULL,1,&dev,NULL,NULL,&e);
    cl_command_queue q=clCreateCommandQueue(ctx,dev,CL_QUEUE_PROFILING_ENABLE,&e);
    char*src=rd(clp);const char*s[]={src};
    cl_program pr=clCreateProgramWithSource(ctx,1,s,NULL,&e);
    if(clBuildProgram(pr,1,&dev,"",NULL,NULL)!=CL_SUCCESS){size_t ls;clGetProgramBuildInfo(pr,dev,CL_PROGRAM_BUILD_LOG,0,NULL,&ls);char*lg=(char*)malloc(ls+1);clGetProgramBuildInfo(pr,dev,CL_PROGRAM_BUILD_LOG,ls,lg,NULL);lg[ls]=0;fprintf(stderr,"%s\n",lg);return 1;}
    cl_ulong l2=0;clGetDeviceInfo(dev,CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,sizeof(l2),&l2,NULL);
    printf("L2 informada pelo driver: %.0f KB\n\n",(double)l2/1024.0);

    const int GS=65536, R=512, RUNS=25;
    const char*ks[]={"sweep_noflag","sweep_glc","sweep_glc_slc"};
    const char*fl[]={"GLC=0 SLC=0","GLC=1 SLC=0","GLC=1 SLC=1"};
    struct { const char* nome; size_t nelem; } casos[] = {
        { "A) 1 MB   — CABE na L2 (2 MB)",      262144u   },
        { "B) 64 MB  — NAO cabe na L2",         16777216u }
    };
    for(int c=0;c<2;c++){
        size_t nelem=casos[c].nelem, bytes=nelem*sizeof(float);
        float*hA=(float*)malloc(bytes);
        for(size_t i=0;i<nelem;i++)hA[i]=1.0f;
        cl_mem dA=clCreateBuffer(ctx,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,bytes,hA,&e);
        cl_mem dO=clCreateBuffer(ctx,CL_MEM_WRITE_ONLY,GS*sizeof(float),NULL,&e);
        printf("=== %s  (buffer %.0f MB) ===\n",casos[c].nome,(double)bytes/1048576.0);
        printf("%-14s %-12s %10s %12s %10s\n","kernel","flags","mediana(ms)","GB/s","vs GLC=1");
        double base=0;
        for(int k=0;k<3;k++){
            cl_kernel kn=clCreateKernel(pr,ks[k],&e);
            cl_int mask=(cl_int)(nelem-1),rr=R,gs=GS;
            clSetKernelArg(kn,0,sizeof(cl_mem),&dA);clSetKernelArg(kn,1,sizeof(cl_mem),&dO);
            clSetKernelArg(kn,2,sizeof(cl_int),&mask);clSetKernelArg(kn,3,sizeof(cl_int),&rr);
            clSetKernelArg(kn,4,sizeof(cl_int),&gs);
            size_t g=GS,l=256;double t[64];
            for(int r=0;r<RUNS+3;r++){
                cl_event ev;clEnqueueNDRangeKernel(q,kn,1,NULL,&g,&l,0,NULL,&ev);
                clWaitForEvents(1,&ev);cl_ulong a,b;
                clGetEventProfilingInfo(ev,CL_PROFILING_COMMAND_START,sizeof(a),&a,NULL);
                clGetEventProfilingInfo(ev,CL_PROFILING_COMMAND_END,sizeof(b),&b,NULL);
                if(r>=3)t[r-3]=(b-a)/1e6; clReleaseEvent(ev);
            }
            qsort(t,RUNS,sizeof(double),cmpd);
            double med=t[RUNS/2];
            double gb=((double)GS*R*4.0)/(med/1000.0)/1e9;
            if(k==1)base=med;
            printf("%-14s %-12s %10.4f %12.1f %10s\n",ks[k],fl[k],med,gb,
                   k==2?"":"-");
            if(k==2)printf("%-14s %-12s %10s %12s %9.3fx\n","","","","",med/base);
            clReleaseKernel(kn);
        }
        printf("\n");
        clReleaseMemObject(dA);clReleaseMemObject(dO);free(hA);
    }
    return 0;
}
