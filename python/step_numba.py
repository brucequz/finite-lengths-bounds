import numpy as np
from numba import cuda
import math


@cuda.jit
def accumulate_to_spectrum(buffer, state_idx, spectrum):
    """Accumulates the specific row of the buffer into the spectrum stored on device.

    Args:
        buffer: complete distance spectrum of all states and stages.
        state_idx: the beginning state needed to be extracted from buffer.
        spectrum: the output. A small array of actual TBCC spectrum stored on device.

    """
    x = cuda.grid(1)
    max_X = buffer.shape[1]

    if x < max_X:
        cuda.atomic.add(spectrum, x, buffer[state_idx, x])


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
    num_states = A_shape[0]
    curr_max_weight = A_shape[1]

    x, y, z = cuda.grid(3)
    bx, by, bz = cuda.blockIdx.x, cuda.blockIdx.y, cuda.blockIdx.z
    bs_x, bs_y, bs_z = cuda.blockDim.x, cuda.blockDim.y, cuda.blockDim.z
    tx, ty, tz = cuda.threadIdx.x, cuda.threadIdx.y, cuda.threadIdx.z
    W_shape, D_shape = W_in.shape, D_in.shape
    mid_y = num_states // 2

    ## Load W into shared memory
    # we process 64 states for each thread block
    # storage requirement: 64*2*1 = 128 bytes
    shared_W = cuda.shared.array(
        shape=(32, 2), dtype=np.uint8
    )  # 1 bytes for weight, support up to weight=255
    # set shared_W to 0 before using
    if ty < bs_y and z < W_shape[1]:
        shared_W[ty, tz] = 0
        shared_W[ty + bs_y, tz] = 0
    # set W
    if y < mid_y and ty < bs_y and z < W_shape[1]:
        shared_W[ty, tz] = W_in[y, z]
        shared_W[ty + bs_y, tz] = W_in[y + mid_y, z]

    ## Load D into shared memory
    # we process 64 states for each thread block
    # storage requirement: 64*2*4 = 512 bytes
    shared_D = cuda.shared.array(
        shape=(32, 2), dtype=np.uint32
    )  # 32 bits to support states up to 2^32-1
    if ty < bs_y and z < D_shape[1]:
        shared_D[ty, tz] = 0
        shared_D[ty + bs_y, tz] = 0
    if y < mid_y and ty < bs_y and z < D_shape[1]:
        shared_D[ty, tz] = D_in[y, z]
        shared_D[ty + bs_y, tz] = D_in[y + mid_y, z]

    num_blk_iters = math.ceil(curr_max_weight / bs_x)
    for i_blkiter in range(num_blk_iters):
        x_id = x + i_blkiter * bs_x

        ## Load A into shared memory
        # storage requirement: 64*32*8 = 16,384 or 64*32*16 = 32,768
        shared_A = cuda.shared.array(
            shape=(32, 32), dtype=np.uint64
        )  # 64 bits or 128 bits
        if ty < bs_y and tx < bs_x:
            shared_A[ty, tx] = 0
            shared_A[ty + bs_y, tx] = 0
        if y < mid_y and ty < bs_y and x_id < curr_max_weight:
            shared_A[ty, tx] = A_in[y, x_id]
            shared_A[ty + bs_y, tx] = A_in[y + mid_y, x_id]

        ## Allocate space for output in shared memory
        shared_out = cuda.shared.array(shape=(32, 34), dtype=np.uint64)
        if ty < bs_y and tx < bs_x:
            shared_out[ty, tx] = 0
            shared_out[ty + bs_y, tx] = 0
        if ty < bs_y and tx < 2:
            shared_out[ty, tx + bs_x] = 0
            shared_out[ty + bs_y, tx + bs_x] = 0

        cuda.syncthreads()

        dst_state_offset = shared_D[0, 0]

        ## first group of states shift and write to result
        if y < mid_y and ty < bs_y and x_id < A_shape[1]:
            shift_amt_0 = shared_W[ty, tz]
            dst_state_0 = shared_D[ty, tz] - dst_state_offset
            shifted_x_0 = tx + shift_amt_0
            shared_out[dst_state_0, shifted_x_0] += shared_A[ty, tx]

        cuda.syncthreads()

        ## second group of states shift and write to result
        if y < mid_y and ty < bs_y and x_id < A_shape[1]:
            shift_amt_1 = shared_W[ty + bs_y, tz]
            dst_state_1 = shared_D[ty + bs_y, tz] - dst_state_offset
            shifted_x_1 = tx + shift_amt_1
            shared_out[dst_state_1, shifted_x_1] += shared_A[ty + bs_y, tx]

        cuda.syncthreads()

        ## Write shared_out to global memory
        # To avoid atomic operations, each threadblock needs to process
        # the entire row of A_in
        if x_id < curr_max_weight + 2:
            out[2 * y + z, x_id] += shared_out[2 * ty + tz, tx]
            if tx < 2 and x_id + bs_x < curr_max_weight + 2:
                out[2 * y + z, x_id + bs_x] += shared_out[2 * ty + tz, tx + bs_x]

        cuda.syncthreads()
