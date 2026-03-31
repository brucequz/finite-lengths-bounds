import numpy as np
from numba import cuda


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


@cuda.jit
def numba_sharedMem_trellisStep_shift(A_in, A_shape, W_in, D_in, out):
    """
    Computes trellis Step for one meta-stage. The previous distance spectrum is stored in A_in.
    The transition weight matrix for one meta-stage is stored in W_in with ending state queried from D_in.
    The required W_in and A_in segments are loaded in the shared memory.

    The first step is for each state to query the current distance spectrum from A_in, then query the weight
    added with the current step from W_in. Each thread block should process 32 length in x-dimension (weight spectrum)
    and 32 states, and 1 in the number of inputs dimension. Overall, the shared memory required
    for A_in is 32*32*1*8 = 8192 bytes for int64 and 32*32*1*16 = 16384 bytes for int128. The shared memory required for W_in
    is 32*1*1 = 32 bytes. The shared memory required for D_in is 32*1*4 = 128 bytes.

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
    W_shape = W_in.shape
    D_shape = D_in.shape

    ## Load W into shared memory
    shared_W = cuda.shared.array(
        shape=(32, 1), dtype=np.uint8
    )  # 1 bytes for weight, support up to weight=255
    if y < W_shape[0] and z < W_shape[1]:
        shared_W[ty, tz] = W_in[y, z]
    cuda.syncthreads()

    ## Load D into shared memory
    shared_D = cuda.shared.array(
        shape=(32, 1), dtype=np.uint32
    )  # 32 bits to support states up to 2^32-1
    if y < D_shape[0] and z < D_shape[1]:
        shared_D[ty, tz] = D_in[y, z]
    cuda.syncthreads()

    ## Load A into shared memory
    shared_A = cuda.shared.array(
        shape=(32, 32), dtype=np.float64
    )  # 64 bits or 128 bits
    if y < A_shape[0] and x < A_shape[1]:
        shared_A[ty, tx] = A_in[y, x]
    cuda.syncthreads()

    # With each thread block, figure out the shift_amt and destination_state
    shift_amt = shared_W[ty, tz]
    dst_state = shared_D[ty, tz]

    # find global output write location
    shifted_x = x + shift_amt  # if x = 10
    if y < A_shape[0] and x < A_shape[1]:
        cuda.atomic.add(out, (dst_state, shifted_x), shared_A[ty, tx])


@cuda.jit
def numba_sharedMem_trellisStep_foldshift(A_in, A_shape, W_in, D_in, out):
    """Computes trellisStep using fold shift method with shared & constant memory.

    Fold shift means we process two input states at the same time; since for a single output
    state, there are two input states i.e., both input state 000 and 100 lead to output state 001 if input=1.
    This method processes pairs of such states together to avoid atomic operations. For each output state,
    there are two input states that map to it.

    The dimension of each thread block is [x,y,z], with x being the weight-direction, y being the states-direction, and
    z being the input-direction.

    Let each thread block process n pairs of such states, so the total number of input states processed by this block is 2n.
    The simple way is to load in the entire spectrum for its responsible states into shared memory at once, then
    process it in smaller chunks in x-direction. Let's say each thread block processes n output states, then it needs to
    take in the first n states from the first half of the input distance spectrum, and the first n states from
    the second half of the input distance spectrum. These two segments can be loaded into the shared
    memory.

    Since we have processed all possible input states for any output state, we can accumulate the result within
    a single thread block and store the result temporarily in shared memory before writing back to global memory.

    For now, we assign the x-dimension of the thread block to be 32, so that each warp performs consecutive memory access.

    steps:
        1. bring in both segments of A_in to shared memory, A_g1 (group1) and A_g2 (group2).
        2. bring in both segments of W_in and D_in to constant/shared memory.
        3. allocate space for out in shared memory, shared_out.
        4. processes chunks of group1 and group2 sequentially and write to shared_out.
        5. Write back to global ping-pong buffers upon finishing processing all A_g1 and A_g2.

    """
    # TODO: implement in NUMBA
    pass
