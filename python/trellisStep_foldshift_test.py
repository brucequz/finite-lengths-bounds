import numpy as np
import yaml
import math
from numba import cuda
from setup import setup_A_Wbit_D
from step import trellisStep_shift, trellisStep_folded_shift
from step_numba import numba_sharedMem_trellisStep_foldshift, accumulate_to_spectrum

import sys


def main():

    path = "config/k51n126v8.yaml"
    try:
        with open(path, "r") as f:
            code_config = yaml.safe_load(f)
        print(f"Successfully loaded: {path}")
    except FileNotFoundError:
        print(f"Error: The file '{path}' was not found.")
        sys.exit(1)
    output_file_name = code_config["output_file_name"]

    As, W_weight, D, basis, num_trellis_stages = setup_A_Wbit_D(path)
    A = As[10]
    W_y, W_z = W_weight.shape
    A_shape = A.shape
    O_y, O_x = A_shape
    print("W_weight: ", W_weight)

    max_shift_per_stage = 2

    # DUT
    # prepare ping-pong buffer at CPU
    num_streams = len(As)
    cuda_streams = [cuda.stream() for _ in range(num_streams)]

    max_X = A_shape[1] + (max_shift_per_stage) * num_trellis_stages
    h_in_buffer = np.zeros(shape=(O_y, max_X), dtype=np.float64)
    h_out_buffer = np.zeros(shape=(O_y, max_X), dtype=np.float64)
    # initialize the distance spectrum on GPU ONCE
    d_spectrum = cuda.to_device(np.zeros(max_X))

    for i_stream, A in enumerate(As):

        A_shape = A.shape
        O_y, O_x = A_shape

        # assign current stream
        curr_stream = cuda_streams[i_stream]

        h_in_buffer[0 : A.shape[0], 0 : A.shape[1]] = A

        # move ping-pong buffers to GPU
        d_buffer_in = cuda.to_device(h_in_buffer)
        d_buffer_out = cuda.to_device(h_out_buffer)
        d_buffer_allzero = cuda.to_device(h_out_buffer)

        # moving W and D to GPU
        d_W = cuda.to_device(W_weight)
        d_D = cuda.to_device(D)

        # block and grid size allocation
        threads_per_block = (32, 32, 1)

        for i_stage in range(num_trellis_stages):
            O_x += max_shift_per_stage

            grid_x = 1
            grid_y = math.ceil((O_y // 2) / threads_per_block[1])
            grid_z = math.ceil(W_z / threads_per_block[2])
            blocks_per_grid = (grid_x, grid_y, grid_z)

            # Zero the entire max buffer
            d_buffer_out.copy_to_device(d_buffer_allzero)

            numba_sharedMem_trellisStep_foldshift[blocks_per_grid, threads_per_block](
                d_buffer_in, A_shape, d_W, d_D, d_buffer_out
            )

            d_buffer_in, d_buffer_out = d_buffer_out, d_buffer_in

            A_shape = tuple((O_y, O_x))

        # Accumulate to spectrum on GPU
        threads_per_block_1d = 256
        grid_size_1d = math.ceil(max_X / threads_per_block_1d)

        accumulate_to_spectrum[grid_size_1d, threads_per_block_1d, curr_stream](
            d_buffer_in, basis[i_stream], d_spectrum
        )

    # synchronize all streams
    cuda.synchronize()

    gpu_distance_spectrum = d_spectrum.copy_to_host()
    print("gpu_distance_spectrum: ", gpu_distance_spectrum)
    np.save("output/fold/" + output_file_name, gpu_distance_spectrum)


def test_spectrum():
    path = "config/k21n62v6.yaml"
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
    W_y, W_z = W_weight.shape
    A_shape = A.shape
    O_y, O_x = A_shape
    print("W_weight: ", W_weight)

    num_stages = 30
    max_shift_per_stage = 2

    ## Ref
    # a single trellis Step
    ref_result = A
    for i_stage in range(num_stages):
        ref_result = trellisStep_shift(ref_result, W_weight, D, max_shift_per_stage)

    # DUT
    # prepare ping-pong buffer at CPU
    max_X = A_shape[1] + (max_shift_per_stage) * num_stages
    h_in_buffer = np.zeros(shape=(O_y, max_X), dtype=np.float64)
    h_in_buffer[0 : A.shape[0], 0 : A.shape[1]] = A
    h_out_buffer = np.zeros(shape=(O_y, max_X), dtype=np.float64)

    # move ping-pong buffers to GPU
    d_buffer_in = cuda.to_device(h_in_buffer)
    d_buffer_out = cuda.to_device(h_out_buffer)
    d_buffer_allzero = cuda.to_device(h_out_buffer)

    # moving W and D to GPU
    d_W = cuda.to_device(W_weight)
    d_D = cuda.to_device(D)

    # block and grid size allocation
    threads_per_block = (32, 32, 1)

    for i_stage in range(num_stages):
        print("i_stage: ", i_stage)
        O_x += max_shift_per_stage

        grid_x = 1
        grid_y = math.ceil((O_y // 2) / threads_per_block[1])
        grid_z = math.ceil(W_z / threads_per_block[2])
        blocks_per_grid = (grid_x, grid_y, grid_z)
        print("blocks_per_grid: ", blocks_per_grid)

        # Zero the entire max buffer
        d_buffer_out.copy_to_device(d_buffer_allzero)

        numba_sharedMem_trellisStep_foldshift[blocks_per_grid, threads_per_block](
            d_buffer_in, A_shape, d_W, d_D, d_buffer_out
        )

        d_buffer_in, d_buffer_out = d_buffer_out, d_buffer_in

        A_shape = tuple((O_y, O_x))

    dut_result = d_buffer_in.copy_to_host()

    assert ref_result.shape == dut_result.shape
    print("ref_result shape: ", ref_result.shape)
    print("dut_result shape: ", dut_result.shape)

    if np.allclose(ref_result, dut_result):
        print("Success!")
    else:
        diff_indices = np.argwhere(ref_result != dut_result)
        print(diff_indices)


if __name__ == "__main__":
    main()
