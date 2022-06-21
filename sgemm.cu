#include <stdlib.h>
#include <algorithm>
#include <cmath>
#include <assert.h>
#include <iostream>

#include <cuda_runtime.h>

#include <curand.h>
#include <cublas_v2.h>

template<int Ms, int Ns, int Ks, int blockDimX, int blockDimY>
__global__ __launch_bounds__(blockDimX * blockDimY)
void sgemm(int M, int N, int K, const float alpha, const float beta,
           const float *A,  const float *B, float *C) {
    // A(M * K), B(K * N), C(M * N)
    // data tile C(Ms * Ns) for each block
    int &lda = M, &ldb = K, &ldc = M;
    
    // diagonal access, for optimizing DRAM access (but seems reducing L2 hit rate)
    // new_block.x = blockIdx.x
    // new_block.y = (blockIdx.x + blockIdx.y) % gridDim.y
    // A += blockIdx.x * Ms;
    // B += (blockIdx.y + blockIdx.x) % gridDim.y * Ns * ldb;
    // C += (blockIdx.y + blockIdx.x) % gridDim.y * Ns * ldc + blockIdx.x * Ms;
    A += blockIdx.x * Ms;
    B += blockIdx.y * Ns * ldb;
    C += blockIdx.y * Ns * ldc + blockIdx.x * Ms;
    
    // shared memory limit occupancy, thus only tileB used double buffer
    __shared__ float4 tileA[Ks][Ms/4],
                      tileB[2][Ns/4][Ks];
    
    constexpr int Mr = Ms / blockDimX,
                  Nr = Ns / blockDimY;
    float accumulator[Nr][Mr] {};
    float fragA[Ms / blockDimX] {};
    float fragB[Ns / blockDimY] {};
    float4 nextA[Mr/4] {};
    float4 nextB[Nr/4] {};
    
    // fill the first tileA tileB
    # pragma unroll
    for (int y = 0; y < Ks; y += blockDimY) {
        # pragma unroll
        for (int x = 0; x < Ms/4; x += blockDimX) {
            tileA[y + threadIdx.y][x + threadIdx.x].x = A[(y + threadIdx.y) * lda + 4*x + threadIdx.x];
            tileA[y + threadIdx.y][x + threadIdx.x].y = A[(y + threadIdx.y) * lda + 4*x + blockDimX + threadIdx.x];
            tileA[y + threadIdx.y][x + threadIdx.x].z = A[(y + threadIdx.y) * lda + 4*x + 2*blockDimX + threadIdx.x];
            tileA[y + threadIdx.y][x + threadIdx.x].w = A[(y + threadIdx.y) * lda + 4*x + 3*blockDimX + threadIdx.x];
        }
    }
    # pragma unroll
    for (int y = 0; y < Ns/4; y += blockDimY) {
        # pragma unroll
        for (int x = 0; x < Ks; x += blockDimX) {
            tileB[0][y + threadIdx.y][x + threadIdx.x].x = B[(4*y + threadIdx.y) * ldb + x + threadIdx.x];
            tileB[0][y + threadIdx.y][x + threadIdx.x].y = B[(4*y + blockDimY + threadIdx.y) * ldb + x + threadIdx.x];
            tileB[0][y + threadIdx.y][x + threadIdx.x].z = B[(4*y + 2*blockDimY + threadIdx.y) * ldb + x + threadIdx.x];
            tileB[0][y + threadIdx.y][x + threadIdx.x].w = B[(4*y + 3*blockDimY + threadIdx.y) * ldb + x + threadIdx.x];
        }
    }
    __syncthreads();
    
    for (int kblock = 0; kblock < (K + Ks - 1) / Ks; kblock += 1) {
        // calculate and accumulate
        // Mr * Ks, Ks * Nr -> Mr * Nr
        // strided access
        # pragma unroll
        for (int kid = 0; kid < Ks; kid ++) {
            // (prefetching next tile) part1
            // load global -> register when a data block start
            if ((kid & (blockDimY - 1)) == 0 && kblock != (K + Ks - 1) / Ks) {
                # pragma unroll
                for (int x = 0; x < Ms/4; x += blockDimX) {
                    int y = kid / blockDimY * blockDimY;
                    nextA[x / blockDimX].x = A[(y + threadIdx.y + Ks) * lda + 4*x + threadIdx.x];
                    nextA[x / blockDimX].y = A[(y + threadIdx.y + Ks) * lda + 4*x + blockDimX + threadIdx.x];
                    nextA[x / blockDimX].z = A[(y + threadIdx.y + Ks) * lda + 4*x + 2*blockDimX + threadIdx.x];
                    nextA[x / blockDimX].w = A[(y + threadIdx.y + Ks) * lda + 4*x + 3*blockDimX + threadIdx.x];
                }
            }
            if ((kid & (blockDimX - 1)) == 0 && kblock != (K + Ks - 1) / Ks) {
                # pragma unroll
                for (int y = 0; y < Ns/4; y += blockDimY) {
                    int x = kid / blockDimX * blockDimX;
                    nextB[y / blockDimY].x = B[(4*y + threadIdx.y) * ldb + x + threadIdx.x + Ks];
                    nextB[y / blockDimY].y = B[(4*y + blockDimY + threadIdx.y) * ldb + x + threadIdx.x + Ks];
                    nextB[y / blockDimY].z = B[(4*y + 2*blockDimY + threadIdx.y) * ldb + x + threadIdx.x + Ks];
                    nextB[y / blockDimY].w = B[(4*y + 3*blockDimY + threadIdx.y) * ldb + x + threadIdx.x + Ks];
                }
            }
            
            # pragma unroll
            for (int x = 0; x < Ms; x += 4*blockDimX) {
                fragA[x / blockDimX] = tileA[kid][x/4 + threadIdx.x].x;
                fragA[x / blockDimX+1] = tileA[kid][x/4 + threadIdx.x].y;
                fragA[x / blockDimX+2] = tileA[kid][x/4 + threadIdx.x].z;
                fragA[x / blockDimX+3] = tileA[kid][x/4 + threadIdx.x].w;
            }
            # pragma unroll
            for (int y = 0; y < Ns; y += 4*blockDimY) {
                fragB[y / blockDimY] = tileB[kblock&1][y/4 + threadIdx.y][kid].x;
                fragB[y / blockDimY+1] = tileB[kblock&1][y/4 + threadIdx.y][kid].y;
                fragB[y / blockDimY+2] = tileB[kblock&1][y/4 + threadIdx.y][kid].z;
                fragB[y / blockDimY+3] = tileB[kblock&1][y/4 + threadIdx.y][kid].w;
            }
            # pragma unroll
            for (int x = 0; x < Ms; x += blockDimX) {
                # pragma unroll
                for (int y = 0; y < Ns; y += blockDimY) {
                    accumulator[y / blockDimY][x / blockDimX] += fragA[x / blockDimX] * fragB[y / blockDimY];
                }
            }

            // (prefetching next tile) part2
            // store register -> shared when a data block end (to hide the LDG latency )
            if ((kid & (blockDimX - 1)) == blockDimX - 1 && kblock != (K + Ks - 1) / Ks) {
                // here useing double tileB to cancel a sync
                # pragma unroll
                for (int y = 0; y < Ns/4; y += blockDimY) {
                    tileB[(kblock+1)&1][y + threadIdx.y][kid / blockDimX * blockDimX + threadIdx.x] = nextB[y / blockDimY];
                }
            }
            if ((kid & (blockDimY - 1)) == blockDimY - 1 && kblock != (K + Ks - 1) / Ks) {
                __syncthreads();
                # pragma unroll
                for (int x = 0; x < Ms/4; x += blockDimX) {
                    tileA[kid / blockDimY * blockDimY + threadIdx.y][x + threadIdx.x] = nextA[x / blockDimX];
                }
            }
        }
        
        A += Ks * lda;
        B += Ks;
        
        // __syncthreads();
    }
    
    // store global
    # pragma unroll
    for (int y = 0; y < Ns; y += blockDimY) {
        # pragma unroll
        for (int x = 0; x < Ms; x += blockDimX) {
            int cid = (y + threadIdx.y) * ldc + x + threadIdx.x;
            C[cid] = beta * C[cid] + alpha * accumulator[y / blockDimY][x / blockDimX];
        }
    }
    
}


template<int Ms, int Ns, int Ks, int blockDimX, int blockDimY>
void launch_sgemm(int M, int N, int K, float alpha, float beta,
                  const float *A,  const float *B, float *C) {
    assert(Ms >= blockDimX);
    assert(Ns >= blockDimY);
    assert(Ks >= blockDimY);
    assert(Ks >= blockDimX);
    // assert(Ks * Ms >= blockDimX * blockDimY);
    // assert(Ks * Ns >= blockDimX * blockDimY);
    assert(Ms / blockDimX >= 4); // for float4
    assert(Ns / blockDimY >= 4);
    sgemm<Ms, Ns, Ks, blockDimX, blockDimY>
        <<<dim3((M + Ms - 1)/Ms, (N + Ns - 1)/Ns), dim3(blockDimX, blockDimY)>>>
        (M, N, K, alpha, beta, A, B, C);
}


int main(int argc, char *argv[]) {
    
    int M, N, K;
    if (argc == 4) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    }
    else if (argc > 1) {
        M = atoi(argv[1]); N = M; K = M;
    }
    else {
        M = 2048; K = 512; N = 2048;
    }
    // cudaDeviceSetCacheConfig(cudaFuncCachePreferNone);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // random matrix
    float *d_A, *d_B, *d_C, *d_C_ref;
    float alpha = 1.0f, beta = 1.0f;
    float time0, time1;
    cudaMalloc((float **)&d_A, M * K * sizeof(float));
    cudaMalloc((float **)&d_B, K * N * sizeof(float));
    cudaMalloc((float **)&d_C, M * N * sizeof(float));
    cudaMalloc((float **)&d_C_ref, M * N * sizeof(float));
    
    const long long seed = 12345;
    curandGenerator_t rand_state;
    curandCreateGenerator(&rand_state, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(rand_state, seed);
    curandGenerateNormal(rand_state, d_A, M * K, 0.0f, 1.0f);
    curandGenerateNormal(rand_state, d_B, K * N, 0.0f, 1.0f);
    curandGenerateNormal(rand_state, d_C, M * N, 0.0f, 1.0f);
    cudaMemcpy(d_C_ref, d_C, M * N * sizeof(float), cudaMemcpyDeviceToDevice);
    
    
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    
    cudaEventRecord(start);
    for (int i = 0; i < 5; i ++)
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
                    &alpha, d_A, M, d_B, K, &beta, d_C_ref, M);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time0, start, stop);
    std::cout << "cublas:   " << time0 << std::endl;
    
    // Ms=128, Ns=64, Ks=16, blockDimX=16, blockDimY=8
    auto launch_sgemm_0 = launch_sgemm<128, 64, 16, 16, 8>;
    cudaEventRecord(start);
    for (int i = 0; i < 5; i ++)
        launch_sgemm_0(M, N, K, alpha, beta, d_A, d_B, d_C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time1, start, stop);
    std::cout << "mykernel: " << time1 << std::endl;
    std::cout << "achieved: " << time0 / time1 << std::endl;
    
    
    // verify
    float *h_C_ref = new float[M * N] {};
    float *h_C = new float[M * N] {};
    cudaMemcpy(h_C_ref, d_C_ref, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    double err_rate {};
    double max_err {};
    for (int i = 0; i < M * N; i ++) {
        double err = std::abs((h_C_ref[i] - h_C[i]) / (std::abs(h_C_ref[i]) + 1e-12));
        if (err > 1e-5) err_rate += 1;
        max_err = std::max(max_err, err);
    }
    std::cout << "max_err: " << max_err << " err_rate: " << err_rate/(M*N) << std::endl;
    
    delete [] h_C_ref;
    delete [] h_C;
    cublasDestroy(cublas_handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_C_ref);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    cudaDeviceReset();
}
