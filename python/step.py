import numpy as np


def trellisStep_conv(A, W, D):

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
        # input
        for y in range(O_y):
            # state
            end_state = int(D[y, c])
            O[c, end_state, :] += np.convolve(A[y, :], W[c, y, :])

    result = np.sum(O, axis=0)
    return result


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


def trellisStep_folded_shift(A, W, D, max_shift):
    """CPU version of the folded shift computation.
    The computation of distance spectrum should be viewed from the output side.
    That is, for each output state, we find the pair of input states that would transition
    to this particular output state. Then we find corresponding rows of W and D matrices to
    compute the resulting otuput spectrum. This should be done by looping over all possible
    output states.

    Args:
        - A: [num_states x max_weight at current stage]
        - W: [num_states x input]
        - D: [num_states x input]
        - max_shift: maximum amount of weight increase in one stage

    Out:
        - newA: [num_states x max_weight after current stage]

    """

    old_ds_shape = A.shape
    num_states = old_ds_shape[0]
    curr_max_weight = old_ds_shape[1]
    new_max_weight = curr_max_weight + max_shift
    newA = np.zeros(shape=(num_states, new_max_weight))

    num_inputs = W.shape[1]

    # process for each output state
    mid_state = num_states // 2
    for i_end_state in range(num_states):
        input = i_end_state % 2
        # first incoming state to this end state
        begin_state_0 = i_end_state // 2
        shift_amt_0 = W[begin_state_0, input]
        shifted_ds = np.zeros(shape=(new_max_weight,))
        shifted_ds[shift_amt_0 : shift_amt_0 + curr_max_weight] = A[begin_state_0, :]
        newA[i_end_state, :] += shifted_ds

        # second incoming state to this end state
        begin_state_1 = begin_state_0 + mid_state
        shift_amt_1 = W[begin_state_1, input]
        shifted_ds = np.zeros(shape=(new_max_weight,))
        shifted_ds[shift_amt_1 : shift_amt_1 + curr_max_weight] = A[begin_state_1, :]
        newA[i_end_state, :] += shifted_ds

    return newA
