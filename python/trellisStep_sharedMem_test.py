import math
import numpy as np
import yaml
import os, sys
from numba import cuda
from setup import setup_A_Wbit_D
from step import trellisStep_shift
from step_numba import numba_sharedMem_trellisStep_shift
import ctypes

# 1. Load the Shared Library
lib_path = os.path.abspath("lib/libtrellis.so")
cuda_lib = ctypes.CDLL(lib_path)

# Setup argument types
cuda_lib.launch_trellis_kernel.argtypes = [
    ctypes.c_int,  # max_weight
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,  # A
    ctypes.c_void_p,
    ctypes.c_int,  # W
    ctypes.c_void_p,  # D
    ctypes.c_void_p,
    ctypes.c_int,  # Out
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,  # Grid
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,  # Block
]


def run_cuda_trellis(max_weight, d_A, A_cols, d_W, d_D, d_out, out_cols):
    # 1. Extract raw integer addresses from Numba DeviceNDArrays
    # .device_ctypes_pointer.value gives the actual 64-bit memory address
    ptr_A = d_A.device_ctypes_pointer.value
    ptr_W = d_W.device_ctypes_pointer.value
    ptr_D = d_D.device_ctypes_pointer.value
    ptr_out = d_out.device_ctypes_pointer.value

    # 2. Force dimensions to standard Python ints
    A_rows, _ = map(int, d_A.shape)
    W_cols = int(d_W.shape[1])

    # 3. Grid/Block logic
    bx, by, bz = 32, 32, 1
    gx = (A_cols + bx - 1) // bx
    gy = (A_rows + by - 1) // by
    gz = (W_cols + bz - 1) // bz

    # 4. Call the C++ function
    # Pass the .value (integers) directly; ctypes.c_void_p handles the conversion
    cuda_lib.launch_trellis_kernel(
        max_weight,
        ptr_A,
        A_rows,
        A_cols,
        ptr_W,
        W_cols,
        ptr_D,
        ptr_out,
        out_cols,
        int(gx),
        int(gy),
        int(gz),
        int(bx),
        int(by),
        int(bz),
    )


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

    num_stages = 1
    max_shift_per_stage = 2
    overall_max_shift = num_stages * max_shift_per_stage

    ## Ref
    ref_result = A
    for i_stage in range(num_stages):
        ref_result = trellisStep_shift(ref_result, W_weight, D, max_shift_per_stage)
    print("ref_result shape: ", ref_result.shape)
    print("ref_result: \n", ref_result)

    ## DUT1
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

    ## CUDA DUT
    A_shape = A.shape
    W_y, W_z = W_weight.shape
    print("max_X: ", max_X)

    # output shape
    O_x = A_shape[1]
    for iter in range(num_stages):
        # Calculate the output width
        O_x += max_shift_per_stage

        # Dynamic Grid Calculation (based on logical output width)
        grid_x = math.ceil(O_x / threads_per_block[0])
        grid_y = math.ceil(O_y / threads_per_block[1])
        grid_z = math.ceil(W_z / threads_per_block[2])
        blocks_per_grid = (grid_x, grid_y, grid_z)

        # Zero the entire max buffer
        d_buffer_out.copy_to_device(d_buffer_allzero, stream=curr_stream)

        # Kernel Launch
        run_cuda_trellis(max_X, d_buffer_in, A_shape[1], d_W, d_D, d_buffer_out, O_x)

        # 7. The Swap (Variables now point to the other buffer)
        d_buffer_in, d_buffer_out = d_buffer_out, d_buffer_in

        # 8. Update the logical width for the NEXT iteration
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
