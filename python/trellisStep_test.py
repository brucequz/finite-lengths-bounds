import sys
import yaml
import math
import numpy as np
from numba import cuda
from setup import setup_A_Wbit_D, computeMetaStage
from step_numba import numba_sharedMem_trellisStep_shift
import argparse


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

    As, W, D, basis, num_trellis_stages = setup_A_Wbit_D(path)

    # compute meta stage
    length_meta_stage = 1
    metaW, metaD = W, D
    for i_meta in range(int(np.log2(length_meta_stage))):
        print("i_meta=", i_meta)
        print("metaW shape: ", metaW.shape)
        print("metaD shape: ", metaD.shape)
        metaW, metaD = computeMetaStage(metaW, metaD)
        metaW = np.ascontiguousarray(metaW)
        metaD = np.ascontiguousarray(metaD)
    W_y, W_z = W.shape
    metaW_y, metaW_z = metaW.shape
    print(f"metaW shape: {metaW.shape}; metaD shape: {metaD.shape}")

    num_stages = 1
    max_shift_per_leftover_stage = 2
    max_shift_per_metastage = length_meta_stage * max_shift_per_leftover_stage
    overall_max_shift = num_stages * max_shift_per_leftover_stage

    num_leftover_iters = num_trellis_stages % length_meta_stage
    num_meta_iters = num_trellis_stages // length_meta_stage
    print("num_leftover_iters: ", num_leftover_iters)
    print("num_meta_iters: ", num_meta_iters)

    # ## Ref
    # ref_result = []
    # for A in As:
    #     cpu_result = A
    #     for iter in range(num_meta_iters):
    #         # a meta-stage trellis Step
    #         cpu_result = trellisStep_shift(
    #             cpu_result, metaW, metaD, max_shift_per_metastage
    #         )
    #     ref_result.append(cpu_result)
    # print("len(ref_result): ", len(ref_result))
    # cpu_distance_spectrum = np.zeros_like(ref_result[0][0])
    # for i_vss, valid_starting_state in enumerate(basis):
    #     cpu_distance_spectrum += ref_result[i_vss][valid_starting_state, :]
    # print("cpu_distance_spectrum: ", cpu_distance_spectrum)

    ## DUT
    # output shape
    num_streams = len(As)
    cuda_streams = [cuda.stream() for _ in range(num_streams)]

    # prepare ping-pong buffer ONCE at CPU
    A_shape = As[0].shape
    max_X = (
        A_shape[1]
        + max_shift_per_leftover_stage * num_leftover_iters
        + max_shift_per_metastage * num_meta_iters
    )
    h_in_buffer = np.zeros(shape=(A_shape[0], max_X), dtype=np.float64)

    h_out_buffer = np.zeros(shape=(A_shape[0], max_X), dtype=np.float64)

    # initialize the distance spectrum on GPU ONCE
    d_spectrum = cuda.to_device(np.zeros(max_X))

    for i_stream, A in enumerate(As):

        A_shape = A.shape
        O_x = A_shape[1]
        O_y = A_shape[0]

        # assign current stream
        curr_stream = cuda_streams[i_stream]

        # Assign values in CPU buffer
        h_in_buffer[0 : A_shape[0], 0 : A_shape[1]] = A

        # move ping-pong buffers to GPU
        d_buffer_in = cuda.to_device(h_in_buffer, stream=curr_stream)
        d_buffer_out = cuda.to_device(h_out_buffer, stream=curr_stream)
        d_buffer_allzero = cuda.to_device(h_out_buffer, stream=curr_stream)

        # moving W and D to GPU
        d_W = cuda.to_device(W, stream=curr_stream)
        d_metaW = cuda.to_device(metaW, stream=curr_stream)
        d_D = cuda.to_device(D, stream=curr_stream)
        d_metaD = cuda.to_device(metaD, stream=curr_stream)

        # block and grid size allocation
        threads_per_block = (32, 32, 1)  # x, y, z

        ## Leftover iterations
        for lo_iter in range(num_leftover_iters):

            O_x += max_shift_per_leftover_stage

            grid_x = math.ceil(O_x / threads_per_block[0])
            grid_y = math.ceil(O_y / threads_per_block[1])
            grid_z = math.ceil(W_z / threads_per_block[2])
            blocks_per_grid = (grid_x, grid_y, grid_z)

            # leftover single-stage steps
            numba_sharedMem_trellisStep_shift[
                blocks_per_grid, threads_per_block, curr_stream
            ](d_buffer_in, A_shape, d_W, d_D, d_buffer_out)

            # swap the first step result to d_buffer_in
            d_buffer_in, d_buffer_out = d_buffer_out, d_buffer_in

            # Update input shape after leftover step
            A_shape = tuple((A_shape[0], O_x))

        ## Meta iterations
        for iter in range(num_meta_iters):
            # Calculate the output width
            O_x += max_shift_per_metastage

            # 3. Dynamic Grid Calculation (based on logical output width)
            grid_x = math.ceil(O_x / threads_per_block[0])
            grid_y = math.ceil(O_y / threads_per_block[1])
            grid_z = math.ceil(W_z / threads_per_block[2])
            blocks_per_grid = (grid_x, grid_y, grid_z)

            # Zero the entire max buffer
            d_buffer_out.copy_to_device(d_buffer_allzero, stream=curr_stream)

            # Kernel Launch
            numba_sharedMem_trellisStep_shift[
                blocks_per_grid, threads_per_block, curr_stream
            ](d_buffer_in, A_shape, d_metaW, d_metaD, d_buffer_out)

            # 7. The Swap (Variables now point to the other buffer)
            d_buffer_in, d_buffer_out = d_buffer_out, d_buffer_in

            # 8. Update the logical width for the NEXT iteration
            A_shape = tuple((A_shape[0], O_x))

        # Accumulate to spectrum on GPU
        threads_per_block_1d = 256
        grid_size_1d = math.ceil(max_X / threads_per_block_1d)

        accumulate_to_spectrum[grid_size_1d, threads_per_block_1d, curr_stream](
            d_buffer_in, basis[i_stream], d_spectrum
        )

        # # Copy result back to CPU
        # dut_result.append(d_buffer_in.copy_to_host(stream=curr_stream))

    # synchronize all streams
    cuda.synchronize()

    gpu_distance_spectrum = d_spectrum.copy_to_host()
    print("gpu_distance_spectrum: ", gpu_distance_spectrum)
    np.save("output/" + output_file_name, gpu_distance_spectrum)

    # ## Check
    # assert len(ref_result) == len(dut_result)
    # for i_out in range(len(ref_result)):
    #     assert ref_result[i_out].shape == dut_result[i_out].shape

    #     if np.allclose(dut_result[i_out], ref_result[i_out]):
    #         # print(
    #         #     f"Success stream {i_out}!",
    #         # )
    #         pass
    #     else:
    #         diff = np.abs(dut_result[i_out] - ref_result[i_out])
    #         max_diff = np.max(diff)
    #         print(f"The maximum difference is: {max_diff}")
    #         idx = np.unravel_index(np.argmax(diff), diff.shape)
    #         print(f"Location of max difference: {idx}")


if __name__ == "__main__":
    main()
