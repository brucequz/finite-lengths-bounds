import os, sys
import yaml
import math
import numpy as np
from numba import cuda
from setup import setup_A_Wbit_D, computeMetaStage
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

    path = "config/k15n30v6.yaml"
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

    max_shift_per_leftover_stage = 2
    max_shift_per_metastage = length_meta_stage * max_shift_per_leftover_stage

    num_leftover_iters = num_trellis_stages % length_meta_stage
    num_meta_iters = num_trellis_stages // length_meta_stage
    print("num_leftover_iters: ", num_leftover_iters)
    print("num_meta_iters: ", num_meta_iters)

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
    h_in_buffer = np.zeros(shape=(A_shape[0], max_X), dtype=np.uint64)
    h_out_buffer = np.zeros(shape=(A_shape[0], max_X), dtype=np.uint64)

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
            run_cuda_trellis(
                max_X, d_buffer_in, A_shape[1], d_W, d_D, d_buffer_out, O_x
            )

            # swap the first step result to d_buffer_in
            d_buffer_in, d_buffer_out = d_buffer_out, d_buffer_in

            # Update input shape after leftover step
            A_shape = tuple((A_shape[0], O_x))

        ## Meta iterations
        for iter in range(num_meta_iters):
            # Calculate the output width
            O_x += max_shift_per_metastage

            # Dynamic Grid Calculation (based on logical output width)
            grid_x = math.ceil(O_x / threads_per_block[0])
            grid_y = math.ceil(O_y / threads_per_block[1])
            grid_z = math.ceil(W_z / threads_per_block[2])
            blocks_per_grid = (grid_x, grid_y, grid_z)

            # Zero the entire max buffer
            d_buffer_out.copy_to_device(d_buffer_allzero, stream=curr_stream)

            # Kernel Launch
            run_cuda_trellis(
                max_X, d_buffer_in, A_shape[1], d_W, d_D, d_buffer_out, O_x
            )

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

    # synchronize all streams
    cuda.synchronize()

    gpu_distance_spectrum = d_spectrum.copy_to_host()
    print("gpu_distance_spectrum: ", gpu_distance_spectrum)
    np.save("output/" + output_file_name, gpu_distance_spectrum)


if __name__ == "__main__":
    main()
