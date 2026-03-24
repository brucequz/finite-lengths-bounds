import math
import numpy as np
from numba import cuda
from step import numba_sharedMem_trellisStep_shift


def trellisStep(A, W, D):

    A_x = A.shape[1]
    A_y = A.shape[0]

    W_y = W.shape[0]
    W_z = W.shape[1]

    D_y = D.shape[0]
    D_x = D.shape[1]

    # for every 4 states, add a different number to it
    increments = np.arange(W_y) // 8
    result = W + increments[:, None]

    return result


def main():

    num_streams = 1
    cuda_streams = [cuda.stream() for _ in range(num_streams)]
    rng = np.random.default_rng(seed=42)
    # create input A and transition matrix W and end-state matrix D
    A_x = 5
    A_y = 8
    A = rng.integers(low=0, high=6, size=(A_y, A_x)).astype(np.float32)
    # print("A:", A)
    W_x = 32
    W_y = A_y
    W = rng.integers(low=0, high=6, size=(W_y, W_x)).astype(np.float32)
    print("W:", W)
    D_x = W_x
    D_y = A_y
    D = rng.integers(low=0, high=D_y, size=(D_y, D_x)).astype(np.float32)
    # print("D:", D)

    num_iters = 1

    ## Ref
    ref_result = trellisStep(A, W, D)
    print("ref_result: ", ref_result)

    ## DUT
    # output shape
    O_x = A_x + W_x - 1
    O_y = A_y

    for i_stream in range(num_streams):

        # assign current stream
        curr_stream = cuda_streams[i_stream]

        # prepare ping-pong buffer at CPU
        max_X = A.shape[1] + (W_x - 1) * num_iters
        h_in_buffer = np.zeros(shape=(O_y, max_X), dtype=np.float64)
        h_in_buffer[0 : A.shape[0], 0 : A.shape[1]] = A
        h_out_buffer = np.zeros(shape=(O_y, max_X), dtype=np.float64)

        # move ping-pong buffers to GPU
        d_buffer_in = cuda.to_device(h_in_buffer, stream=curr_stream)
        d_buffer_out = cuda.to_device(h_out_buffer, stream=curr_stream)
        d_buffer_allzero = cuda.to_device(h_out_buffer, stream=curr_stream)

        # moving W and D to GPU
        d_W = cuda.to_device(W, stream=curr_stream)
        d_D = cuda.to_device(D, stream=curr_stream)

        # block and grid size allocation
        threads_per_block = (32, 8, 2)

        grid_x = math.ceil(O_x / threads_per_block[0])
        grid_y = math.ceil(O_y / threads_per_block[1])
        grid_z = math.ceil(W_x / threads_per_block[2])
        blocks_per_grid = (grid_x, grid_y, grid_z)

        numba_sharedMem_trellisStep_shift[
            blocks_per_grid, threads_per_block, curr_stream
        ](d_buffer_in, h_in_buffer.shape, d_W, d_D, d_buffer_out)

    # synchronize all streams
    cuda.synchronize()

    dut_result = d_W.copy_to_host()
    print("dut_result: ", dut_result)

    ## Check
    assert ref_result.shape == dut_result.shape

    if np.allclose(dut_result, ref_result):
        print(
            f"Success!",
        )
    else:
        diff = np.abs(dut_result - ref_result)
        max_diff = np.max(diff)
        print(f"The maximum difference is: {max_diff}")
        idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"Location of max difference: {idx}")


if __name__ == "__main__":
    main()
