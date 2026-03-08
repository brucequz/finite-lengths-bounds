#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>

namespace py = pybind11;

__global__ void square_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * data[idx];
    }
}

__global__ void merge(const int* A1, const int* A2, int* A_merge, int dim2, int num_states, int n_s_1, int n_s_2) {

    int sb = blockIdx.x * blockDim.x + threadIdx.x;
    int se = blockIdx.y * blockDim.y + threadIdx.y;
    int k  = blockIdx.z * blockDim.z + threadIdx.z;

    int n_s_merge = n_s_1 + n_s_2 - 1;

    if (sb < num_states && se < num_states && k < n_s_merge) {
        int tmp_sum = 0;

        for (int i = 0; i < num_states; ++i) {
            for (int j = 0; j < n_s_1; j++) {
                int k_minus_j = k - j;
                
                if (k_minus_j >= 0 && k_minus_j < n_s_2) {
                    // A1 index: [sb, i, j]
                    int idx1 = sb * (num_states * n_s_1) + i * n_s_1 + j;
                    // A2 index: [i, se, k-j]
                    int idx2 = i * (num_states * n_s_2) + se * n_s_2 + k_minus_j;

                    tmp_sum += A1[idx1] * A2[idx2];
                }
            }
        }
        
        // A_merge index: [sb, se, k]
        int idx_out = sb * (num_states * n_s_merge) + se * n_s_merge + k;
        A_merge[idx_out] = tmp_sum;
    }
}

py::array_t<int> run_merge_gpu(py::array_t<int> A1_py, py::array_t<int> A2_py) {
    // 1. Request buffer info to get pointers and shapes
    auto buf1 = A1_py.request();
    auto buf2 = A2_py.request();

    // Extract dimensions (Assuming A1: [sb, num_states, n_s_1], A2: [num_states, se, n_s_2])
    int dim0 = buf1.shape[0];        // sb
    int num_states = buf1.shape[1];  // shared dim
    int n_s_1 = buf1.shape[2];
    int n_s_2 = buf2.shape[2];
    int n_s_merge = n_s_1 + n_s_2 - 1;

    printf("printing buf1.shape[0]: %d\n", buf1.shape[0]);
    fflush(stdout); // This forces the text to the screen immediately

    size_t size1 = buf1.size * sizeof(int);
    size_t size2 = buf2.size * sizeof(int);
    size_t size_out = (size_t)dim0 * num_states * n_s_merge * sizeof(int);

    // 2. Allocate Device Memory
    int *d_A1, *d_A2, *d_A_merge;
    cudaMalloc(&d_A1, size1);
    cudaMalloc(&d_A2, size2);
    cudaMalloc(&d_A_merge, size_out);

    // 3. Copy Inputs from Host to Device
    cudaMemcpy(d_A1, buf1.ptr, size1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A2, buf2.ptr, size2, cudaMemcpyHostToDevice);

    // 4. Set up Grid/Block dimensions
    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks((dim0 + 7) / 8, 
                   (num_states + 7) / 8, 
                   (n_s_merge + 7) / 8);

    // 5. Launch Kernel
    merge<<<numBlocks, threadsPerBlock>>>(d_A1, d_A2, d_A_merge, n_s_merge, num_states, n_s_1, n_s_2);

    // 6. Prepare Output Array (Python side)
    auto result = py::array_t<int>({dim0, num_states, n_s_merge});
    auto buf_out = result.request();

    // 7. Copy Result from Device to Host
    cudaMemcpy(buf_out.ptr, d_A_merge, size_out, cudaMemcpyDeviceToHost);

    // 8. Cleanup
    cudaFree(d_A1);
    cudaFree(d_A2);
    cudaFree(d_A_merge);

    return result;
}

PYBIND11_MODULE(cuda_kernels, m) {
    m.def("square_array", &run_merge_gpu, "C++20/CUDA Square function");
}