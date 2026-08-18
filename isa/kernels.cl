/* Cenario 1: Acesso normal — L1 -> L2 -> VRAM (GLC=0, SLC=0) */
__kernel void matmul_via_L1(
    __global const float* restrict A,
    __global const float* restrict B,
    __global float*       restrict C,
    const int N)
{
    int row = get_global_id(1);
    int col = get_global_id(0);
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

/* Cenario 2: volatile nos ponteiros — NAO gera GLC=1 no GFX8 (encoding DC500000, identico ao Cenario 1) */
__kernel void matmul_via_L2(
    __global volatile const float* A,
    __global volatile const float* B,
    __global volatile float*       C,
    const int N)
{
    int row = get_global_id(1);
    int col = get_global_id(0);
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

/* Cenario 3: volatile (mesmo que Cenario 2) — NAO gera GLC/SLC no GFX8 (ISA identica ao Cenario 1) */
__kernel void matmul_direto_VRAM(
    __global volatile const float* A,
    __global volatile const float* B,
    __global volatile float*       C,
    const int N)
{
    int row = get_global_id(1);
    int col = get_global_id(0);
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

/* BONUS: Tiled com local memory */
#define TILE_SIZE 16
__kernel void matmul_tiled(
    __global const float* restrict A,
    __global const float* restrict B,
    __global float*       restrict C,
    const int N)
{
    int row = get_global_id(1);
    int col = get_global_id(0);
    int localRow = get_local_id(1);
    int localCol = get_local_id(0);

    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        int tiledCol = t * TILE_SIZE + localCol;
        int tiledRow = t * TILE_SIZE + localRow;

        if (row < N && tiledCol < N)
            tileA[localRow][localCol] = A[row * N + tiledCol];
        else
            tileA[localRow][localCol] = 0.0f;

        if (tiledRow < N && col < N)
            tileB[localRow][localCol] = B[tiledRow * N + col];
        else
            tileB[localRow][localCol] = 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[localRow][k] * tileB[k][localCol];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}
/* Cenario 4: Bypass L1 FORCADO via inline asm — flat_load com glc (GLC=1) */
__kernel void matmul_glc_asm(
    __global const float* restrict A,
    __global const float* restrict B,
    __global float* restrict C,
    const int N)
{
    int row = get_global_id(1);
    int col = get_global_id(0);
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            float a, b;
            __global const float* pa = &A[row * N + k];
            __global const float* pb = &B[k * N + col];

            // flat_load_dword com modificador glc => GLC=1 => contorna L1, le da L2
            __asm__ volatile(
                "flat_load_dword %0, %1 glc\n"
                "s_waitcnt vmcnt(0)\n"
                : "=v"(a)
                : "v"(pa));
            __asm__ volatile(
                "flat_load_dword %0, %1 glc\n"
                "s_waitcnt vmcnt(0)\n"
                : "=v"(b)
                : "v"(pb));

            sum += a * b;
        }
        C[row * N + col] = sum;
    }
}
