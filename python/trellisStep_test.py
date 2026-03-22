import sys
import yaml
import math
import numpy as np
from numba import cuda
from setup import setup_A_W_D, computeMetaStage
from step import numba_trellisStep_conv, trellisStep_conv
import argparse


def main():

    path = "config/k11n30v6.yaml"
    try:
        with open(path, "r") as f:
            code_config = yaml.safe_load(f)
        print(f"Successfully loaded: {path}")
    except FileNotFoundError:
        print(f"Error: The file '{path}' was not found.")
        sys.exit(1)
    output_file_name = code_config["output_file_name"]

    As, W, D, basis, num_trellis_stages = setup_A_W_D(path)

    # compute meta stage
    length_meta_stage = 4
    metaW, metaD = W, D
    for i_meta in range(int(np.log2(length_meta_stage))):
        print("i_meta=", i_meta)
        print("metaW shape: ", metaW.shape)
        print("metaD shape: ", metaD.shape)
        metaW, metaD = computeMetaStage(metaW, metaD)
        metaW = np.ascontiguousarray(metaW)
        metaD = np.ascontiguousarray(metaD)
    W_z, W_y, W_x = W.shape
    metaW_z, metaW_y, metaW_x = metaW.shape

    num_meta_iters = num_trellis_stages // length_meta_stage
    print("num_meta_iters: ", num_meta_iters)

    print(f"metaW shape: {metaW.shape}; metaD shape: {metaD.shape}")

    ## Ref
    ref_result = []
    for A in As:
        # a single trellis Step
        cpu_result = trellisStep_conv(A, W, D)
        for iter in range(num_meta_iters):
            # a meta-stage trellis Step
            cpu_result = trellisStep_conv(cpu_result, metaW, metaD)
        ref_result.append(cpu_result)
    print("len(ref_result): ", len(ref_result))

    ## DUT
    # output shape
    dut_result = []
    num_streams = len(As)
    cuda_streams = [cuda.stream() for _ in range(num_streams)]

    for i_stream, A in enumerate(As):

        # output shape for the leftover step
        A_shape = A.shape
        O_x = A_shape[1] + W_x - 1
        O_y = W_y
        O_z = W_z

        # assign current stream
        curr_stream = cuda_streams[i_stream]

        # prepare ping-pong buffer at CPU
        max_X = A.shape[1] + W_x - 1 + (metaW_x - 1) * num_meta_iters
        h_in_buffer = np.zeros(shape=(O_y, max_X), dtype=np.uint64)
        h_in_buffer[0 : A.shape[0], 0 : A.shape[1]] = A
        h_out_buffer = np.zeros(shape=(O_y, max_X), dtype=np.uint64)

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
        threads_per_block = (8, 16, 2)

        grid_x = math.ceil(O_x / threads_per_block[0])
        grid_y = math.ceil(O_y / threads_per_block[1])
        grid_z = math.ceil(W_z / threads_per_block[2])
        blocks_per_grid = (grid_x, grid_y, grid_z)

        ## leftover single-stage steps
        numba_trellisStep_conv[blocks_per_grid, threads_per_block, curr_stream](
            d_buffer_in, A_shape, d_W, d_D, d_buffer_out
        )

        # swap the first step result to d_buffer_in
        d_buffer_in, d_buffer_out = d_buffer_out, d_buffer_in

        # Update input shape after leftover step
        A_shape = tuple((A_shape[0], O_x))

        for iter in range(num_meta_iters):
            # Calculate the output width
            current_O_x = A_shape[1] + metaW_x - 1
            O_y = metaW_y
            O_z = metaW_z

            # 3. Dynamic Grid Calculation (based on logical output width)
            grid_x = math.ceil(current_O_x / threads_per_block[0])
            grid_y = math.ceil(O_y / threads_per_block[1])
            grid_z = math.ceil(O_z / threads_per_block[2])
            blocks_per_grid = (grid_x, grid_y, grid_z)

            # Zero the entire max buffer
            d_buffer_out.copy_to_device(d_buffer_allzero, stream=curr_stream)

            # Kernel Launch
            numba_trellisStep_conv[blocks_per_grid, threads_per_block, curr_stream](
                d_buffer_in, A_shape, d_metaW, d_metaD, d_buffer_out
            )

            # 7. The Swap (Variables now point to the other buffer)
            d_buffer_in, d_buffer_out = d_buffer_out, d_buffer_in

            # 8. Update the logical width for the NEXT iteration
            A_shape = tuple((A_shape[0], current_O_x))

        dut_result.append(d_buffer_in.copy_to_host(stream=curr_stream))

    # synchronize all streams
    cuda.synchronize()

    print("len(dut_result): ", len(dut_result))

    ## Check
    assert len(ref_result) == len(dut_result)
    for i_out in range(len(ref_result)):
        assert ref_result[i_out].shape == dut_result[i_out].shape

        if np.allclose(dut_result[i_out], ref_result[i_out]):
            print(
                f"Success stream {i_out}!",
            )
        else:
            diff = np.abs(dut_result[i_out] - ref_result[i_out])
            max_diff = np.max(diff)
            print(f"The maximum difference is: {max_diff}")
            idx = np.unravel_index(np.argmax(diff), diff.shape)
            print(f"Location of max difference: {idx}")


if __name__ == "__main__":
    main()
