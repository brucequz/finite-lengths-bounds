import numpy as np
import yaml
from setup import setup_A_Wbit_D
from step import trellisStep_shift, trellisStep_folded_shift

import sys


def main():

    path = "config/k51n126v6.yaml"
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

    ## Ref
    # a single trellis Step
    ref_result = trellisStep_shift(A, W_weight, D, 2)

    # DUT
    dut_result = trellisStep_folded_shift(A, W_weight, D, 2)

    assert ref_result.shape == dut_result.shape
    print("ref_result: ", ref_result)
    print("dut_result: ", dut_result)

    if np.allclose(ref_result, dut_result):
        print("Success!")


if __name__ == "__main__":
    main()
