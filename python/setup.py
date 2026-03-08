import numpy as np
import numpy as np
import yaml
import argparse
import sys
from build import cuda_kernels
from cu_dsu import run_numba_merge


def octal_to_binary_list(octal_str):
    # Convert octal string to integer, then to a binary string
    # '0b' prefix is sliced off using [2:]
    binary_str = bin(int(octal_str, 8))[2:]

    # Convert the string '101' into a list of integers [1, 0, 1]
    return [int(bit) for bit in binary_str]


def bin2dec(binary):
    """
    Note: bin_list is intentionally not reversed to read the number as 'right-msb'.
    It can be "reversed" if the next states is intended to be read as 'left-msb'.
    """
    return [
        int(sum(val * (2**idx) for idx, val in enumerate(bin_list)))
        for bin_list in binary
    ]


def main():

    # 1. Set up the argument parser
    parser = argparse.ArgumentParser(description="Process some YAML configuration.")

    # 2. Add the argument for the config file
    parser.add_argument(
        "config_path",
        help="Path to the yaml configuration file (e.g., config/k11n30v6.yaml)",
    )

    # 3. Parse the arguments from the terminal
    args = parser.parse_args()

    try:
        with open(args.config_path, "r") as f:
            code_config = yaml.safe_load(f)
        print(f"Successfully loaded: {args.config_path}")
    except FileNotFoundError:
        print(f"Error: The file '{args.config_path}' was not found.")
        sys.exit(1)

    nu = code_config["tbcc_config"]["V"]
    m = code_config["bch_config"]["M"]
    K = code_config["bch_config"]["K"]
    crc = code_config["bch_config"]["polynomial"]
    num_concat_memory = nu + m
    num_total_states = 2 ** (num_concat_memory)
    num_valid_starting_states = 2 ** (code_config["tbcc_config"]["V"])
    states = np.arange(0, num_total_states, dtype=np.int32)
    num_trellis_stages = code_config["bch_config"]["K"] + code_config["bch_config"]["M"]
    cwd_max_weight = 2 * num_trellis_stages

    p1 = np.flip(octal_to_binary_list(code_config["tbcc_config"]["gen_poly_1"]))
    p2 = np.flip(octal_to_binary_list(code_config["tbcc_config"]["gen_poly_2"]))
    p_crc = np.flip([int(bit) for bit in code_config["bch_config"]["polynomial"]])

    # convolution between gen_poly with crc
    poly1 = np.mod(np.convolve(p1, p_crc, mode="full"), 2)
    poly2 = np.mod(np.convolve(p2, p_crc, mode="full"), 2)

    # states
    states_str = [np.binary_repr(s, width=num_concat_memory) for s in states]
    flipped_states = [s[::-1] for s in states_str]
    states_matrix = np.array([[int(bit) for bit in s] for s in flipped_states])

    # input, dst states
    v_zeros = np.zeros(shape=(num_total_states, 1))
    v_ones = np.ones(shape=(num_total_states, 1))
    input_0 = np.hstack((v_zeros, states_matrix))
    input_1 = np.hstack((v_ones, states_matrix))
    dst_0 = bin2dec(input_0[:, :num_concat_memory])
    dst_1 = bin2dec(input_1[:, :num_concat_memory])

    # output
    out0 = np.mod(np.matmul(input_0, np.transpose(np.vstack((poly1, poly2)))), 2)
    out1 = np.mod(np.matmul(input_1, np.transpose(np.vstack((poly1, poly2)))), 2)
    Wout0 = np.sum(out0, axis=1, dtype=np.int32)
    Wout1 = np.sum(out1, axis=1, dtype=np.int32)
    zidx0 = (Wout0 == 0).reshape(-1, 1)
    zidx1 = (Wout1 == 0).reshape(-1, 1)

    ## proto distance spectrum
    # [::-1] for the binary representation is for
    # weight 0 -> [1 0 0]; weight 1 -> [0 1 0]; weight 2 -> [0 0 1]
    Wout0_str = [np.binary_repr(weight, width=2)[::-1] for weight in Wout0]
    Wout1_str = [np.binary_repr(weight, width=2)[::-1] for weight in Wout1]
    Wcoef0 = np.concatenate(
        (zidx0, np.array([[int(bit) for bit in s] for s in Wout0_str])), axis=1
    )
    Wcoef1 = np.concatenate(
        (zidx1, np.array([[int(bit) for bit in s] for s in Wout1_str])), axis=1
    )

    basis = np.arange(0, num_total_states, 2 ** code_config["bch_config"]["M"])

    num_workers = 8
    WBTBCCK1 = np.zeros(shape=(num_trellis_stages, num_workers, cwd_max_weight + 1))
    WBTBCCK2 = np.zeros(shape=(num_trellis_stages, num_workers, cwd_max_weight + 1))
    num_valid_starting_states_per_worker = int(num_valid_starting_states / num_workers)

    # fill in A_1
    A_1 = np.zeros(shape=(num_total_states, num_total_states, 3))
    for i_b in np.arange(0, num_total_states):
        A_1[i_b, dst_0[i_b], :] = Wcoef0[i_b, :]
        A_1[i_b, dst_1[i_b], :] = Wcoef1[i_b, :]

    print("A_1 finished")
    print(A_1.shape)

    A_2 = run_numba_merge(A_1, A_1)
    print("A_2 finished")
    A_3 = run_numba_merge(A_1, A_2)
    print("A_3 finished")
    A_4 = run_numba_merge(A_2, A_2)
    print("A_4 finished")
    A_8 = run_numba_merge(A_4, A_4)
    print("A_8 finished")

    # Send to GPU and get result back
    dut_result = cuda_kernels.square_array(A_8, A_4)
    # print("dut_result: ", dut_result)
    print("dut_result shape: ", dut_result.shape)

    A_12 = run_numba_merge(A_8, A_4)
    # print("A_2: ", A_2)
    print("A_2 shape: ", A_2.shape)

    verdict = np.allclose(dut_result, A_12)

    print(f"final verdict:  {verdict}")


if __name__ == "__main__":
    main()
