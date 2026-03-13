import numpy as np
import scipy.linalg as la
import yaml
import argparse
import sys


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


def setup_A_W_D(path):

    try:
        with open(path, "r") as f:
            code_config = yaml.safe_load(f)
        print(f"Successfully loaded: {path}")
    except FileNotFoundError:
        print(f"Error: The file '{path}' was not found.")
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
    dst_0 = np.array(bin2dec(input_0[:, :num_concat_memory]))
    dst_1 = np.array(bin2dec(input_1[:, :num_concat_memory]))

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

    # set up A
    # A_in: [num_states] x [max weight up to this meta-stage]
    basis = np.arange(0, num_total_states, 2 ** code_config["bch_config"]["M"])
    As = []
    for valid_starting_state in basis:
        A = np.zeros(shape=(num_total_states, 1), dtype=np.float64)
        A[valid_starting_state] = 1
        As.append(A)
    print("As length: ", len(As))

    # set up W
    # W_in: [input] x [num_states] x [max weight for one meta-stage]
    W = np.stack((Wcoef0, Wcoef1), axis=0).astype(np.float64)  # horizontal stack
    print("W.shape: ", W.shape)

    # set up D
    # D_in: [num_states] x [input]
    D = np.stack((dst_0, dst_1), axis=1).astype(np.float64)
    print("D shape: ", D.shape)

    return As, W, D, basis, num_trellis_stages
