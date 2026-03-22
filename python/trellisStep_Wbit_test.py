import numpy as np
import yaml
from setup import setup_A_W_D, setup_A_Wbit_D, computeMetaStage
from step import trellisStep_conv
import sys


def trellisStep_shift(ds, W, D, max_shift):
    """Performs a trellis step using the shift method. A new ds variable is returned.

    Args:
        - ds: [num_states x max_weight] distance spectrum upto this trellis stage.
        - W: [num_states x input] for each input and state, there is only 1 number
        representing the weight of the trellis path.
        - D: [num_states x input] destination state for each pair of input and incoming
        state.
        - max_shift: indicator for the increase in output ds dimension.

    Output:
        - newds: [num_states x (max_weight + max_shift)] distance spectrum after this trellis stage.
    """

    old_ds_shape = ds.shape
    num_states = old_ds_shape[0]
    curr_max_weight = old_ds_shape[1]
    new_max_weight = curr_max_weight + max_shift
    newds = np.zeros(shape=(num_states, new_max_weight))

    num_inputs = W.shape[1]

    # process for each input and incoming state
    for i_begin_state in range(num_states):
        for i_input in range(num_inputs):
            shift_amt = W[i_begin_state, i_input]
            dst_state = D[i_begin_state, i_input]
            shifted_ds = np.zeros(shape=(new_max_weight,))
            shifted_ds[shift_amt : shift_amt + curr_max_weight] = ds[i_begin_state, :]
            newds[dst_state, :] += shifted_ds

    return newds


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

    As, W_weight, D, basis, num_trellis_stages = setup_A_Wbit_D(path)
    As, W, D, basis, num_trellis_stages = setup_A_W_D(path)
    A = As[5]

    print("A shape: ", A.shape)
    print("W shape: ", W.shape)
    print("W_weight shape: ", W_weight.shape)
    print("D shape: ", D.shape)

    ## Ref
    # a single trellis Step
    cpu_result = A
    for iter in range(num_trellis_stages):
        # a meta-stage trellis Step
        cpu_result = trellisStep_conv(cpu_result, W, D)
    print("conv result shape: ", cpu_result.shape)

    ## DUT
    num_stages = 1
    max_shift = 2 * num_stages
    for i_step in range(num_trellis_stages):
        A = trellisStep_shift(A, W_weight, D, max_shift)
    print("final A shape: ", A.shape)

    if np.allclose(A, cpu_result):
        print("Success!")


if __name__ == "__main__":
    main()
