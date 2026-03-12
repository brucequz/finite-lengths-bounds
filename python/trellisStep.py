import math
import numpy as np
from numba import cuda


@cuda.jit
def numba_trellisStep(A_in, A_shape, W_in, out):

    x, y, z = cuda.grid(3)

    if z < W_in.shape[0] and y < W_in.shape[1] and x < (W_in.shape[2] + A_shape[1] - 1):
        tmp_sum = 0.0
        for j in range(A_shape[1]):
            x_minus_j = x - j
            if x_minus_j >= 0 and x_minus_j < W_in.shape[2]:
                tmp_sum += W_in[z, y, x_minus_j] * A_in[y, j]

        cuda.atomic.add(out, (y, x), tmp_sum)


# @cuda.jit
# def sum_z_loop_kernel


def trellisStep(A, W):

    A_x = A.shape[1]
    A_y = A.shape[0]

    W_z = W.shape[0]
    W_y = W.shape[1]
    W_x = W.shape[2]

    O_z = W_z
    O_y = W_y
    O_x = A_x + W_x - 1
    O = np.zeros(shape=(O_z, O_y, O_x))
    for c in range(O_z):
        for y in range(O_y):
            O[c, y, :] = np.convolve(A[y, :], W[c, y, :])

    result = np.sum(O, axis=0)
    return result


def main():

    rng = np.random.default_rng(seed=42)
    # create input A and transition matrix W
    A_x = 5
    A_y = 8
    A = rng.integers(low=0, high=6, size=(A_y, A_x))
    W_x = 16
    W_y = A_y
    W_z = 4
    W = rng.integers(low=0, high=6, size=(W_z, W_y, W_x))

    num_iters = 2

    ## Ref
    ref_result = A
    for iter in range(num_iters):
        ref_result = trellisStep(ref_result, W)
    print("ref_result shape: ", ref_result.shape)

    ## DUT
    # output shape
    O_z = W_z
    O_y = W_y

    # prepare ping-pong buffer at CPU
    max_X = A.shape[1] + (W_x - 1) * num_iters
    h_in_buffer = np.zeros(shape=(O_y, max_X), dtype=np.int64)
    h_in_buffer[0 : A.shape[0], 0 : A.shape[1]] = A
    h_out_buffer = np.zeros(shape=(O_y, max_X), dtype=np.int64)

    # move ping-pong buffers to GPU
    d_buffer_in = cuda.to_device(h_in_buffer)
    d_buffer_out = cuda.to_device(h_out_buffer)

    # moving W to GPU
    d_W = cuda.to_device(W)

    # block and grid size allocation
    threads_per_block = (8, 8, 2)

    A_shape = A.shape
    for iter in range(num_iters):
        # Calculate the output width
        current_O_x = A_shape[1] + W_x - 1

        # 3. Dynamic Grid Calculation (based on logical output width)
        grid_x = math.ceil(current_O_x / threads_per_block[0])
        grid_y = math.ceil(O_y / threads_per_block[1])
        grid_z = math.ceil(O_z / threads_per_block[2])
        blocks_per_grid = (grid_x, grid_y, grid_z)

        # Zero the entire max buffer
        d_buffer_out.copy_to_device(h_out_buffer)

        # Kernel Launch
        numba_trellisStep[blocks_per_grid, threads_per_block](
            d_buffer_in, A_shape, d_W, d_buffer_out
        )

        # 7. The Swap (Variables now point to the other buffer)
        d_buffer_in, d_buffer_out = d_buffer_out, d_buffer_in

        # 8. Update the logical width for the NEXT iteration
        A_shape = tuple((A_shape[0], current_O_x))

    dut_result = d_buffer_in.copy_to_host()
    print("dut_result shape: ", dut_result.shape)
    # print(dut_result)

    ## Check
    if np.allclose(dut_result, ref_result):
        print("Success!")


if __name__ == "__main__":
    main()
