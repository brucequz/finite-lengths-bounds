import math
import numpy as np
from numba import cuda


@cuda.jit
def numba_sharedMem_trellisStep(A_in, A_shape, W_in, D_in, out):
    """
    Computes trellis Step for one meta-stage. The previous distance spectrum is stored in A_in.
    The transition matrix for one meta-stage is stored in W_in with ending state queried from D_in.
    The required W_in and A_in segments are loaded in the shared memory.

    The total number of states are separated into 2 segments, the first 2^(nu-1) states begin with a 0
    in the MSB, while the second segment begin with a 1 in the MSB. We note that pairs of states from
    those two segments arrive at the same ending state after the input. For example, state 000 and 100
    both arrive at 000 after input = 0, and 001 after input = 1. Therefore, let's say the thread block
    contains only 2 states in the y-direction, then we would load in A_in with state 000 and 100 into
    shared memory. Correspondingly, the W_in for state 000 and 100 would be loaded to shared memory.


    Args:
        A_in: [num_states] x [max weight up to this meta-stage]
        W_in: [input] x [num_states] x [max weight for one meta-stage]
        D_in: [num_states] x [input]

    Out:
        out: [num_states] x [max weight after this meta-stage]

    z is along input, y is along num_states, and x is along weight
    """

    x, y, z = cuda.grid(3)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    tz = cuda.threadIdx.z

    blockIdx_y = cuda.blockIdx.y  # group idx that we are handling in this block
    blockDim_x = cuda.blockDim.x
    blockDim_y = cuda.blockDim.y  # number of states we are handling in this block
    blockDim_z = cuda.blockDim.z
    W_shape = W_in.shape

    # Load W into shared memory
    # the size should be [:] x [blockDim_y] x [:]
    # the index should be [:, blockIdx_y * blockDim_y : (blockIdx_y+1) * blockDim_y, :]
    shared_W = cuda.shared.array(shape=(2, 4, 8), dtype=np.float32)
    if tz < W_shape[0] and y < W_shape[1] and tx < W_shape[2]:
        shared_W[tz, ty, tx] = W_in[z, y, x]
    cuda.syncthreads()

    # Modify W
    if tz < W_shape[0] and y < W_shape[1] and tx < W_shape[2]:
        shared_W[tz, ty, tx] += blockIdx_y
    cuda.syncthreads()

    # Move W from shared memory to global
    if tz < W_shape[0] and y < W_shape[1] and tx < W_shape[2]:
        W_in[z, y, x] = shared_W[tz, ty, tx]

    # Load A into shared memory
    # the size should be [blockDim_y] x [:]
    # the index should be segment1: [blockIdx_y * (blockDim_y) : (blockIdx_y+1) * (blockDim_y) , :]
    shared_A = cuda.shared.array(shape=(4, 8), dtype=np.float32)
    if y < A_shape[0] and x < A_shape[1]:
        shared_A[ty, tx] = A_in[y, x]
    cuda.syncthreads()

    # # Let's first compute one step, and just verifying the output weight spectrum at state 0
    # if z < W_in.shape[0] and y < W_in.shape[1] and x < (W_in.shape[2] + A_shape[1] - 1):
    #     tmp_sum = 0.0
    #     for j in range(A_shape[1]):
    #         x_minus_j = x - j
    #         if x_minus_j >= 0 and x_minus_j < W_in.shape[2]:
    #             tmp_sum += W_in[z, y, x_minus_j] * A_in[y, j]

    #     # Sum over all possible inputs
    #     end_state = D_in[y, z]
    #     cuda.atomic.add(out, (end_state, x), tmp_sum)


def trellisStep(A, W, D):

    A_x = A.shape[1]
    A_y = A.shape[0]

    W_z = W.shape[0]
    W_y = W.shape[1]
    W_x = W.shape[2]

    D_y = D.shape[0]
    D_x = D.shape[1]

    # for every 4 states, add a different number to it
    increments = np.arange(W_y) // 4
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
    W_x = 16
    W_y = A_y
    W_z = 4
    W = rng.integers(low=0, high=6, size=(W_z, W_y, W_x)).astype(np.float32)
    print("W:", W)
    D_x = W_z
    D_y = A_y
    D = rng.integers(low=0, high=D_y, size=(D_y, D_x)).astype(np.float32)
    # print("D:", D)

    num_iters = 1

    ## Ref
    ref_result = trellisStep(A, W, D)
    print("ref_result: ", ref_result)

    ## DUT
    # output shape
    O_z = W_z
    O_y = W_y

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
        threads_per_block = (8, 4, 2)

        A_shape = A.shape

        for iter in range(num_iters):
            # Calculate the output width
            current_O_x = A_shape[1] + W_x - 1

            # 3. Dynamic Grid Calculation (based on logical output width)
            grid_x = math.ceil(W_x / threads_per_block[0])
            grid_y = math.ceil(O_y / threads_per_block[1])
            grid_z = math.ceil(O_z / threads_per_block[2])
            blocks_per_grid = (grid_x, grid_y, grid_z)
            print("blocks_per_grid: ", blocks_per_grid)

            # Zero the entire max buffer
            d_buffer_out.copy_to_device(d_buffer_allzero, stream=curr_stream)

            # Kernel Launch
            numba_sharedMem_trellisStep[
                blocks_per_grid, threads_per_block, curr_stream
            ](d_buffer_in, A_shape, d_W, d_D, d_buffer_out)

            # 7. The Swap (Variables now point to the other buffer)
            d_buffer_in, d_buffer_out = d_buffer_out, d_buffer_in

            # 8. Update the logical width for the NEXT iteration
            A_shape = tuple((A_shape[0], current_O_x))

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
