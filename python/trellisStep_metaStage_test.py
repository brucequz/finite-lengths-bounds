import sys
import yaml
import math
import numpy as np

# from numba import cuda
from step import numba_trellisStep_conv
from setup import setup_A_W_D, computeMetaStage
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
