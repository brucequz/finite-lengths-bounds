import sys
import yaml
import math
import numpy as np

# from numba import cuda
from setup import setup_A_W_D
import argparse


# @cuda.jit
def numba_metaStage_trellisStep(A_in, A_shape, W_in, D_in, out):
    """
    Computes trellis Step for one meta-stage. The previous distance spectrum is stored in A_in.
    The transition matrix for one meta-stage is stored in W_in with ending state queried from D_in.
    The required W_in segments are loaded in the shared memory.

    The total number of states are separated into 2 segments, the first 2^(nu-1) states begin with a 0
    in the MSB, while the second segment begin with a 1 in the MSB. We note that pairs of states from
    those two segments arrive at the same ending state after the input. For example, state 000 and 100
    both arrive at 000 after input = 0, and 001 after input = 1. Therefore, let's say the thread block
    contains only 2 states in the y-direction, then we would load in A_in with state 000 and 100 into
    shared memory. Correspondingly, the W_in for state 000 and 100 would be loaded to shared memory.


    Args:
        A_in: [num_states] x [max weight up to this meta-stage]
        W_in: [input] x [num_states] x [max weight for one meta-stage]
        D_in: [num_states] x [input]

    Out:
        out: [num_states] x [max weight after this meta-stage]

    z is along input, y is along num_states, and x is along weight
    """

    pass


# @cuda.jit
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
    O = np.zeros(shape=(O_z, O_y, O_x), dtype=np.uint64)
    for c in range(O_z):
        for y in range(O_y):
            end_state = int(D[y, c])
            O[c, end_state, :] += np.convolve(A[y, :], W[c, y, :])

    result = np.sum(O, axis=0)
    return result


def computeMetaStage(W, D, dtype=np.uint64):
    """
    Join num_stages_combine basic stages together into a meta-stage.
    The

    Args:
        - W: [num_inputs, num_states, max_weight with 1 stage]
        - D: [num_states, num_inputs]

    Outs:
        - newW: [num_states, num_states, max_weight with meta-stage]
        - newD: [num_states, 2^num_stages_combine]

    """
    assert W.shape[0] == D.shape[1]
    num_inputs = W.shape[0]
    num_states = W.shape[1]
    curr_max_weight = W.shape[2]

    # Compute newW
    newW = np.zeros(
        shape=(num_states, num_states, 2 * curr_max_weight - 1), dtype=dtype
    )
    newD = np.zeros(shape=(num_states, 2 * num_inputs), dtype=dtype)

    for i_b in range(num_states):
        for i_first_input in range(num_inputs):
            # compute mid state after first meta-stage transition
            mid_state = int(D[i_b, i_first_input])

            for i_second_input in range(num_inputs):
                # compute end state after the second meta-stage transition
                end_state = int(D[mid_state, i_second_input])
                newW[i_b, end_state, :] += np.convolve(
                    W[i_first_input, i_b, :], W[i_second_input, mid_state, :]
                )

                # compute newD
                shift_amt = np.log2(num_inputs)
                concat_input = (int(i_first_input) << int(shift_amt)) | i_second_input
                newD[i_b, concat_input] = end_state

    layer_indices = np.arange(num_states)[:, np.newaxis]
    newW = np.moveaxis(
        newW[layer_indices, newD.astype(np.int32)], source=0, destination=1
    )

    return newW, newD


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
    output_file_name = code_config["output_file_name"]

    As, W, D, basis, num_iters = setup_A_W_D(args.config_path)

    num_iters = 1

    ## Ref
    newW, newD = computeMetaStage(W, D)
    print("newW shape: ", newW.shape)
    print("newW: ", newW[1, :, :])
    print("newD: ", newD[0, :])


if __name__ == "__main__":
    main()
