import math
import numpy as np
import yaml
import sys
from numba import cuda
from setup import setup_A_W_D, setup_A_Wbit_D
from step import trellisStep_shift, numba_sharedMem_trellisStep_shift


# def trellisStep(A, W, D):

#     A_x = A.shape[1]
#     A_y = A.shape[0]

#     W_y = W.shape[0]
#     W_z = W.shape[1]

#     D_y = D.shape[0]
#     D_x = D.shape[1]

#     # for every 4 states, add a different number to it
#     increments = np.arange(W_y) // 8
#     result = W + increments[:, None]

#     return result


def main():

    path = "config/k11n22v3.yaml"
    try:
        with open(path, "r") as f:
            code_config = yaml.safe_load(f)
        print(f"Successfully loaded: {path}")
    except FileNotFoundError:
        print(f"Error: The file '{path}' was not found.")
        sys.exit(1)
    output_file_name = code_config["output_file_name"]

    As, W_weight, D, basis, num_trellis_stages = setup_A_Wbit_D(path)
    A = As[5]

    print("W_weight: \n", W_weight)
    print("D: \n", D)

    num_stages = 10
    max_shift_per_stage = 2
    overall_max_shift = num_stages * max_shift_per_stage

    ## Ref
    ref_result = A
    for i_stage in range(num_stages):
        ref_result = trellisStep_shift(ref_result, W_weight, D, max_shift_per_stage)
    print("ref_result shape: ", ref_result.shape)
    print("ref_result: \n", ref_result)

    ## DUT
    num_streams = 1
    cuda_streams = [cuda.stream() for _ in range(num_streams)]
    A_shape = A.shape
    W_y, W_z = W_weight.shape

    # output shape
    O_x = A_shape[1]
    O_y = A_shape[0]

    for i_stream in range(num_streams):

        # assign current stream
        curr_stream = cuda_streams[i_stream]

        # prepare ping-pong buffer at CPU
        max_X = A_shape[1] + (max_shift_per_stage) * num_stages
        h_in_buffer = np.zeros(shape=(O_y, max_X), dtype=np.float64)
        h_in_buffer[0 : A.shape[0], 0 : A.shape[1]] = A
        h_out_buffer = np.zeros(shape=(O_y, max_X), dtype=np.float64)

        # move ping-pong buffers to GPU
        d_buffer_in = cuda.to_device(h_in_buffer, stream=curr_stream)
        d_buffer_out = cuda.to_device(h_out_buffer, stream=curr_stream)
        d_buffer_allzero = cuda.to_device(h_out_buffer, stream=curr_stream)

        # moving W and D to GPU
        d_W = cuda.to_device(W_weight, stream=curr_stream)
        d_D = cuda.to_device(D, stream=curr_stream)

        # block and grid size allocation
        threads_per_block = (32, 32, 1)

        for i_stage in range(num_stages):
            O_x += max_shift_per_stage

            grid_x = math.ceil(O_x / threads_per_block[0])
            grid_y = math.ceil(O_y / threads_per_block[1])
            grid_z = math.ceil(W_z / threads_per_block[2])
            blocks_per_grid = (grid_x, grid_y, grid_z)

            # Zero the entire max buffer
            d_buffer_out.copy_to_device(d_buffer_allzero, stream=curr_stream)

            numba_sharedMem_trellisStep_shift[
                blocks_per_grid, threads_per_block, curr_stream
            ](d_buffer_in, A_shape, d_W, d_D, d_buffer_out)

            d_buffer_in, d_buffer_out = d_buffer_out, d_buffer_in

            A_shape = tuple((A_shape[0], O_x))

    # synchronize all streams
    cuda.synchronize()

    dut_result = d_buffer_in.copy_to_host()
    print("dut_result shape: ", dut_result.shape)
    print("dut_result: \n", dut_result)

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
