import numpy as np
from numba import cuda


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


@cuda.jit
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


def trellisStep_shift(ds, W, D, max_shift):
    """Performs a trellis step using the shift method. A new ds variable is returned.

    Args:
        - ds: [num_states x max_weight] distance spectrum upto this trellis stage.
        - W: [num_states x input] for each input and state, there is only 1 number
        representing the weight of the trellis path.
        - D: [num_states x input] destination state for each pair of input and incoming
        state.
        - max_shift: indicator for the increase in output ds dimension.

    Output:
        - newds: [num_states x (max_weight + max_shift)] distance spectrum after this trellis stage.
    """

    old_ds_shape = ds.shape
    num_states = old_ds_shape[0]
    curr_max_weight = old_ds_shape[1]
    new_max_weight = curr_max_weight + max_shift
    newds = np.zeros(shape=(num_states, new_max_weight))

    num_inputs = W.shape[1]

    # process for each input and incoming state
    for i_begin_state in range(num_states):
        for i_input in range(num_inputs):
            shift_amt = W[i_begin_state, i_input]
            dst_state = D[i_begin_state, i_input]
            shifted_ds = np.zeros(shape=(new_max_weight,))
            shifted_ds[shift_amt : shift_amt + curr_max_weight] = ds[i_begin_state, :]
            newds[dst_state, :] += shifted_ds

    return newds


@cuda.jit
def numba_sharedMem_trellisStep_shift(A_in, A_shape, W_in, D_in, out):
    """
    Computes trellis Step for one meta-stage. The previous distance spectrum is stored in A_in.
    The transition weight matrix for one meta-stage is stored in W_in with ending state queried from D_in.
    The required W_in and A_in segments are loaded in the shared memory.

    The first step is for each state to query the current distance spectrum from A_in, then query the weight
    added with the current step from W_in. Each thread block should process 32 length in x-dimension (weight spectrum)
    and 8 length in the number of states, and 2 in the number of inputs dimension. Overall, the shared memory required
    for A_in is 32*8*2*8 = 4096 bytes for int64 and 32*8*2*16 = 8192 bytes for int128. The shared memory required for W_in
    is 8*2*1 = 16 bytes. The shared memory required for D_in is 8*2*4 = 64 bytes.

    Args:
        A_in: [num_states] x [max weight up to this meta-stage]
        W_in: [num_states] x [input]
        D_in: [num_states] x [input]

    Out:
        out: [num_states] x [max weight after this meta-stage]

    z is along input, y is along num_states, and x is along weight
    """

    x, y, z = cuda.grid(3)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    tz = cuda.threadIdx.z

    blockIdx_x = cuda.blockIdx.x
    blockIdx_y = cuda.blockIdx.y  # group idx that we are handling in this block
    blockIdx_z = cuda.blockIdx.z
    blockDim_x = cuda.blockDim.x
    blockDim_y = cuda.blockDim.y  # number of states we are handling in this block
    blockDim_z = cuda.blockDim.z
    W_shape = W_in.shape
    D_shape = D_in.shape

    ## Load W into shared memory
    shared_W = cuda.shared.array(
        shape=(8, 2), dtype=np.uint8
    )  # 1 bytes for weight, support up to weight=255
    if y < W_shape[0] and z < W_shape[1]:
        shared_W[ty, tz] = W_in[y, z]
    cuda.syncthreads()

    # # Modify W
    # if ty < W_shape[0] and tz < W_shape[1]:
    #     shared_W[ty, tz] += blockIdx_y
    # cuda.syncthreads()

    # # Move W from shared memory to global
    # if y < W_shape[0] and z < W_shape[1]:
    #     W_in[y, z] = shared_W[ty, tz]
    # cuda.syncthreads()

    ## Load D into shared memory
    shared_D = cuda.shared.array(
        shape=(8, 2), dtype=np.uint32
    )  # 32 bits to support states up to 2^32-1
    if y < D_shape[0] and z < D_shape[1]:
        shared_D[ty, tz] = D_in[y, z]
    cuda.syncthreads()

    ## Load A into shared memory
    shared_A = cuda.shared.array(shape=(32, 8), dtype=np.uint32)  # 64 bits or 128 bits
    if y < A_shape[0] and x < A_shape[1]:
        shared_A[ty, tx] = A_in[y, x]
    cuda.syncthreads()

    # With each thread block, figure out the shift_amt and destination_state
    shift_amt = shared_W[ty, tz]
    dst_state = shared_D[ty, tz]

    # find global output write location
    shifted_x = x + shift_amt
    if shifted_x < out.shape[1]:
        cuda.atomic.add(out, (dst_state, shifted_x), shared_A[ty, tx])
