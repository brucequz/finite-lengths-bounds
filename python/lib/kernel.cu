#include <cuda_runtime.h>
#include <stdint.h>
#include <cstdint>
#include <cstdio>

extern "C" {

    __global__ void trellis_step_kernel(
        int max_weight,
        const unsigned long long* A_in,
        int A_rows,
        int A_cols,
        const uint8_t* W_in,
        int W_cols,
        const uint32_t* D_in,
        unsigned long long* out,
        int out_cols
    ) {
        int x = blockIdx.x * blockDim.x + threadIdx.x; // weight
        int y = blockIdx.y * blockDim.y + threadIdx.y; // states
        int z = blockIdx.z * blockDim.z + threadIdx.z; // input

        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int tz = threadIdx.z;

        // static shared memory allocation
        __shared__ uint8_t shared_W[32][1];
        __shared__ uint32_t shared_D[32][1];
        __shared__ unsigned long long shared_A[32][32];

        // 1. load W into shared memory
        if (y < A_rows && z < W_cols) {
            shared_W[ty][tz] = W_in[y * W_cols + z];
        }

        // 2. load D into shared memory
        if (y < A_rows && z < W_cols) {
            shared_D[ty][tz] = D_in[y * W_cols + z];
        }

        // 3. load A into shared memory
        if (y < A_rows && x < A_cols) {
            shared_A[ty][tx] = A_in[y * max_weight + x];
        }

        __syncthreads();


        // 4. atomic add
        if (y < A_rows && x < A_cols) {
            uint8_t shift_amt = shared_W[ty][tz];
            uint32_t dst_state = shared_D[ty][tz];
            int shifted_x = x + shift_amt;

            int out_idx = dst_state * max_weight + shifted_x;

            atomicAdd(&out[out_idx], shared_A[ty][tx]);

        }
    }

    // DLL Exported function
    void launch_trellis_kernel(
        int max_weight,
        void* A_ptr, int A_rows, int A_cols,
        void* W_ptr, int W_cols,
        void* D_ptr,
        void* out_ptr, int out_cols,
        int gx, int gy, int gz,
        int bx, int by, int bz
    ) {
        dim3 grid(gx, gy, gz);
        dim3 block(bx, by, bz);

        trellis_step_kernel<<<grid, block>>>(
            max_weight,
            (const unsigned long long*)A_ptr, A_rows, A_cols,
            (const uint8_t*)W_ptr, W_cols,
            (const uint32_t*)D_ptr,
            (unsigned long long*)out_ptr, out_cols
        );
    }
}