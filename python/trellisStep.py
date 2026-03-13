import math
import numpy as np
from numba import cuda


@cuda.jit
def numba_trellisStep(A_in, A_shape, W_in, D_in, out):
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


def trellisStep(A, W, D):

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
    O = np.zeros(shape=(O_z, O_y, O_x), dtype=np.float64)
    for c in range(O_z):
        for y in range(O_y):
            end_state = int(D[y, c])
            O[c, end_state, :] += np.convolve(A[y, :], W[c, y, :])

    result = np.sum(O, axis=0)
    return result


def main():

    num_streams = 8
    cuda_streams = [cuda.stream() for _ in range(num_streams)]
    rng = np.random.default_rng(seed=42)
    # create input A and transition matrix W and end-state matrix D
    A_x = 5
    A_y = 64
    As = []
    for _ in range(num_streams):
        As.append(rng.integers(low=0, high=6, size=(A_y, A_x)).astype(np.float64))
    # print("A:", A)
    W_x = 16
    W_y = A_y
    W_z = 4
    W = rng.integers(low=0, high=6, size=(W_z, W_y, W_x)).astype(np.float64)
    # print("W:", W)
    D_x = W_z
    D_y = A_y
    D = rng.integers(low=0, high=D_y, size=(D_y, D_x)).astype(np.float64)
    # print("D:", D)

    num_iters = 50

    ## Ref
    ref_result = []
    for A in As:
        cpu_result = A
        for iter in range(num_iters):
            cpu_result = trellisStep(cpu_result, W, D)
        ref_result.append(cpu_result)
    print("len(ref_result): ", len(ref_result))

    ## DUT
    # output shape
    O_z = W_z
    O_y = W_y
    dut_result = []

    for i_stream, A in enumerate(As):

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

        # moving W and D to GPU
        d_W = cuda.to_device(W, stream=curr_stream)
        d_D = cuda.to_device(D, stream=curr_stream)

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
            d_buffer_out.copy_to_device(h_out_buffer, stream=curr_stream)

            # Kernel Launch
            numba_trellisStep[blocks_per_grid, threads_per_block, curr_stream](
                d_buffer_in, A_shape, d_W, d_D, d_buffer_out
            )

            # 7. The Swap (Variables now point to the other buffer)
            d_buffer_in, d_buffer_out = d_buffer_out, d_buffer_in

            # 8. Update the logical width for the NEXT iteration
            A_shape = tuple((A_shape[0], current_O_x))

        dut_result.append(d_buffer_in.copy_to_host(stream=curr_stream))

    # synchronize all streams
    cuda.synchronize()

    print("len(dut_result): ", len(dut_result))
    print("run finished!")

    ## Check
    assert len(ref_result) == len(dut_result)
    for i_out in range(len(ref_result)):
        assert ref_result[i_out].shape == dut_result[i_out].shape

        if np.allclose(dut_result[i_out], ref_result[i_out]):
            print(
                f"Success stream {i_out}!",
            )
        else:
            diff = np.abs(dut_result - ref_result)
            max_diff = np.max(diff)
            print(f"The maximum difference is: {max_diff}")
            idx = np.unravel_index(np.argmax(diff), diff.shape)
            print(f"Location of max difference: {idx}")


if __name__ == "__main__":
    main()
