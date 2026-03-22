import numpy as np

# from numba import cuda


def trellisStep_conv(A, W, D):

    A_x = A.shape[1]
    A_y = A.shape[0]

    W_z = W.shape[0]
    W_y = W.shape[1]
    W_x = W.shape[2]

    D_y = D.shape[0]
    D_x = D.shape[1]

    O_z = W_z
    O_y = W_y
    O_x = A_x + W_x - 1
    O = np.zeros(shape=(O_z, O_y, O_x), dtype=np.uint64)
    for c in range(O_z):
        # input
        for y in range(O_y):
            # state
            end_state = int(D[y, c])
            O[c, end_state, :] += np.convolve(A[y, :], W[c, y, :])

    result = np.sum(O, axis=0)
    return result


# @cuda.jit
def numba_trellisStep_conv(A_in, A_shape, W_in, D_in, out):
    """
    Computes trellis Step for one meta-stage. The previous distance spectrum is stored in A_in.
    The transition matrix for one meta-stage is stored in W_in with ending state queried from D_in.

    Args:
        A_in: [num_states] x [max weight up to this meta-stage]
        W_in: [input] x [num_states] x [max weight for one meta-stage]
        D_in: [num_states] x [input]

    Out:
        out: [num_states] x [max weight after this meta-stage]

    z is along input, y is along num_states, and x is along weight
    """
    x, y, z = cuda.grid(3)

    if z < W_in.shape[0] and y < W_in.shape[1] and x < (W_in.shape[2] + A_shape[1] - 1):
        tmp_sum = 0.0
        for j in range(A_shape[1]):
            x_minus_j = x - j
            if x_minus_j >= 0 and x_minus_j < W_in.shape[2]:
                tmp_sum += W_in[z, y, x_minus_j] * A_in[y, j]

        # Sum over all possible inputs
        end_state = D_in[y, z]
        cuda.atomic.add(out, (end_state, x), tmp_sum)


# @cuda.jit
def numba_sharedMem_trellisStep_conv(A_in, A_shape, W_in, D_in, out):
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
