import numpy as np
import yaml
from setup import setup_A_W_D, computeMetaStage
import sys


def bitwise_rotate_and_sum(arr, bit_control):
    """
    Shifts the array to the right with zero-padding based on
    the bit positions of bit_control and returns the sum.
    """
    result = np.zeros_like(arr, dtype=np.int64)
    n = len(arr)

    # Iterate through the bits of the control number
    for i in range(bit_control.bit_length()):
        if (bit_control >> i) & 1:
            # We want a shift of 'i'. If your example 0101 means 1 and 3,
            # use shift_amount = i + 1
            shift_amount = i

            if shift_amount == 0:
                result += arr
            elif shift_amount < n:
                # Create a shifted version with zero padding:
                # [0, 0, ... , arr[0], arr[1], ...]
                shifted = np.zeros_like(arr)
                shifted[shift_amount:] = arr[:-shift_amount]
                result += shifted
            # If shift_amount >= n, the result is all zeros, so we do nothing

    return result


def main():
    a = np.array([10, 20, 30, 40])
    control = 5  # 0101
    output = bitwise_rotate_and_sum(a, control)
    print(f"Original: {a}")

    print(f"Result:   {output}")

    path = "config/k11n22v3.yaml"
    try:
        with open(path, "r") as f:
            code_config = yaml.safe_load(f)
        print(f"Successfully loaded: {path}")
    except FileNotFoundError:
        print(f"Error: The file '{path}' was not found.")
        sys.exit(1)
    output_file_name = code_config["output_file_name"]

    As, W, D, basis, num_trellis_stages = setup_A_W_D(path)

    print("W: ", W)

    # compute meta stage
    length_meta_stage = 2
    metaW, metaD = W, D
    for i_meta in range(int(np.log2(length_meta_stage))):
        metaW, metaD = computeMetaStage(metaW, metaD)
        metaW = np.ascontiguousarray(metaW)
        metaD = np.ascontiguousarray(metaD)
    W_z, W_y, W_x = W.shape
    metaW_z, metaW_y, metaW_x = metaW.shape

    print("metaW shape: ", metaW.shape)
    print("metaD shape: ", metaD.shape)
    print("metaW: ", metaW)


if __name__ == "__main__":
    main()
