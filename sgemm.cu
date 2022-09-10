#include <stdlib.h>
#include <algorithm>
#include <cmath>
#include <assert.h>
#include <iostream>

#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>
#include <cudaProfiler.h>

#include <thrust/device_vector.h>


template<int Ms, int Ns, int Ks, int blockDimX, int blockDimY>
__global__ __launch_bounds__(blockDimX * blockDimY)
void sgemm(int M, int N, int K, const float alpha, const float beta,
           const float *A,  const float *B, float *C) {
    // A(M * K), B(K * N), C(M * N)
    // data tile C(Ms * Ns) for each block
    int &lda = M, &ldb = K, &ldc = M;
    
    A += blockIdx.x * Ms;
    B += blockIdx.y * Ns * ldb;
    C += blockIdx.y * Ns * ldc + blockIdx.x * Ms;
    
    // shared memory limit occupancy, thus only tileB used double buffer
    __shared__ float4 tileA[Ks][Ms/4],
                      tileB[2][Ns/4][Ks];
    
    constexpr int Mr = Ms / blockDimX,
                  Nr = Ns / blockDimY;
    float accumulator[Nr][Mr] {};
    float fragA[Mr] {};
    float fragB[Nr] {};
    float4 nextA[Mr/4] {};
    float4 nextB[Nr/4] {};

    auto cp_float_to_float4 = [] __device__ (float4 *tgt, const float *src, int stride) {
            tgt[0].x = src[0];
            tgt[0].y = src[stride];
            tgt[0].z = src[stride*2];
            tgt[0].w = src[stride*3];
    };
    auto cp_float4_to_float = [] __device__ (float *tgt, const float4 *src, int stride) {
        tgt[0] = src[0].x;
        tgt[stride] = src[0].y;
        tgt[stride*2] = src[0].z;
        tgt[stride*3] = src[0].w;
    };
    
    // fill the first tileA tileB
    # pragma unroll
    for (int y = 0; y < Ks; y += blockDimY) {
        # pragma unroll
        for (int x = 0; x < Ms/4; x += blockDimX) {
            cp_float_to_float4(
                &tileA[y + threadIdx.y][x + threadIdx.x],
                &A[(y + threadIdx.y) * lda + 4*x + threadIdx.x],
                blockDimX);
        }
    }
    # pragma unroll
    for (int y = 0; y < Ns/4; y += blockDimY) {
        # pragma unroll
        for (int x = 0; x < Ks; x += blockDimX) {
            cp_float_to_float4(
                &tileB[0][y + threadIdx.y][x + threadIdx.x],
                &B[(4*y + threadIdx.y) * ldb + x + threadIdx.x],
                blockDimY * ldb);
        }
    }
    __syncthreads();
    
    for (int k_stride = 0; k_stride < (K + Ks - 1) / Ks; k_stride ++) {
        // calculate and accumulate
        // strided access
        # pragma unroll
        for (int kid = 0; kid < Ks; kid ++) {
            // (prefetching next tile) part1
            // load global -> register when a data block start
            if ((kid & (blockDimY - 1)) == 0 && k_stride != (K - 1) / Ks) {
                # pragma unroll
                for (int x = 0; x < Ms/4; x += blockDimX) {
                    cp_float_to_float4(
                        &nextA[x / blockDimX],
                        &A[(kid / blockDimY * blockDimY + threadIdx.y + Ks) * lda + 4*x + threadIdx.x],
                        blockDimX);
                }
            }
            if ((kid & (blockDimX - 1)) == 0 && k_stride != (K - 1) / Ks) {
                # pragma unroll
                for (int y = 0; y < Ns/4; y += blockDimY) {
                    cp_float_to_float4(
                        &nextB[y / blockDimY],
                        &B[(4*y + threadIdx.y) * ldb + kid / blockDimX * blockDimX + threadIdx.x + Ks],
                        blockDimY * ldb);
                }
            }
            
            # pragma unroll
            for (int x = 0; x < Ms; x += 4*blockDimX) {
                cp_float4_to_float(&fragA[x / blockDimX], &tileA[kid][x/4 + threadIdx.x], 1);
            }
            # pragma unroll
            for (int y = 0; y < Ns; y += 4*blockDimY) {
                cp_float4_to_float(&fragB[y / blockDimY], &tileB[k_stride&1][y/4 + threadIdx.y][kid], 1);
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
            if ((kid & (blockDimX - 1)) == blockDimX - 1 && k_stride != (K - 1) / Ks) {
                # pragma unroll
                for (int y = 0; y < Ns/4; y += blockDimY) {
                    tileB[(k_stride+1)&1][y + threadIdx.y][kid / blockDimX * blockDimX + threadIdx.x] = nextB[y / blockDimY];
                }
            }
            if ((kid & (blockDimY - 1)) == blockDimY - 1 && k_stride != (K - 1) / Ks) {
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
    assert(Ms / blockDimX >= 4); // for float4
    assert(Ns / blockDimY >= 4);
    sgemm<Ms, Ns, Ks, blockDimX, blockDimY>
        <<<dim3((M + Ms - 1)/Ms, (N + Ns - 1)/Ns), dim3(blockDimX, blockDimY)>>>
        (M, N, K, alpha, beta, A, B, C);
}


int main(int argc, char *argv[]) {
    int M = 2048,
        N = 2048,
        K = 512,
        test_times = 1;
    if (argc == 2) {
        test_times = atoi(argv[1]);
    }
    else if (argc == 4) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    }
    else if (argc == 5) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
        test_times = atoi(argv[4]);
    }
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    
    cuProfilerStop();

    // random matrix
    thrust::device_vector<float> d_A(M * K);
    thrust::device_vector<float> d_B(K * N);
    thrust::device_vector<float> d_C(M * N);
    thrust::device_vector<float> d_C_ref(M * N);
    float *d_A_ptr = thrust::raw_pointer_cast(d_A.data());
    float *d_B_ptr = thrust::raw_pointer_cast(d_B.data());
    float *d_C_ptr = thrust::raw_pointer_cast(d_C.data());
    float *d_C_ref_ptr = thrust::raw_pointer_cast(d_C_ref.data());
    float alpha = 1.0f, beta = 1.0f;
    float time0, time1;
    
    const long long seed = 12345;
    curandGenerator_t rand_state;
    curandCreateGenerator(&rand_state, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(rand_state, seed);
    curandGenerateNormal(rand_state, d_A_ptr, M * K, 0.0f, 1.0f);
    curandGenerateNormal(rand_state, d_B_ptr, K * N, 0.0f, 1.0f);
    curandGenerateNormal(rand_state, d_C_ptr, M * N, 0.0f, 1.0f);
    cudaMemcpy(d_C_ref_ptr, d_C_ptr, M * N * sizeof(float), cudaMemcpyDeviceToDevice);
    
    cuProfilerStart();

    cudaEventRecord(start);
    for (int i = 0; i < test_times; i ++) {
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
                    &alpha, d_A_ptr, M, d_B_ptr, K, &beta, d_C_ref_ptr, M);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time0, start, stop);
    std::cout << "cublas:   " << time0 << std::endl;

    auto launch_sgemm_0 = launch_sgemm<128, 64, 16, 16, 8>;
    cudaEventRecord(start);
    for (int i = 0; i < test_times; i ++) {
        launch_sgemm_0(M, N, K, alpha, beta, d_A_ptr, d_B_ptr, d_C_ptr);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time1, start, stop);
    std::cout << "mykernel: " << time1 << std::endl;
    std::cout << "achieved: " << time0 / time1 << std::endl;
    
    cuProfilerStop();

    // verify
    thrust::host_vector<float> h_C_ref = d_C_ref;
    thrust::host_vector<float> h_C = d_C;
    double err_rate {};
    double max_err {};
    for (int i = 0; i < M * N; i ++) {
        double err = std::abs((h_C_ref[i] - h_C[i]) / (std::abs(h_C_ref[i]) + 1e-12));
        if (err > 1e-5) err_rate += 1;
        max_err = std::max(max_err, err);
    }
    std::cout << "max_err: " << max_err << " err_rate: " << err_rate/(M*N) << std::endl;
    
    cublasDestroy(cublas_handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

