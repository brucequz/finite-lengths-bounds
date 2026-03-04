import numpy as np
import yaml
from scipy.io import loadmat


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


with open("config/k11n30v6.yaml", "r") as f:
    code_config = yaml.safe_load(f)
print(code_config)

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

print(p1)
print(p2)
print(p_crc)
print(octal_to_binary_list(code_config["tbcc_config"]["gen_poly_1"]))
print(octal_to_binary_list(code_config["tbcc_config"]["gen_poly_2"]))

# convolution between gen_poly with crc
poly1 = np.mod(np.convolve(p1, p_crc, mode="full"), 2)
poly2 = np.mod(np.convolve(p2, p_crc, mode="full"), 2)
print("poly1: ", poly1)
print("poly2: ", poly2)

# states
states_str = [np.binary_repr(s, width=num_concat_memory) for s in states]
flipped_states = [s[::-1] for s in states_str]
states_matrix = np.array([[int(bit) for bit in s] for s in flipped_states])
print(states_str[:10])
print(flipped_states[:10])
print(states_matrix[:10, :])

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

print(f"{num_trellis_stages}, {cwd_max_weight + 1}")

for i_worker in range(0, num_workers):
    states_for_worker = basis[
        i_worker
        * num_valid_starting_states_per_worker : (i_worker + 1)
        * num_valid_starting_states_per_worker
    ]
    print(f"i_worker: {i_worker}; states_for_worker: {states_for_worker}")
    for state in states_for_worker:
        # states x weight
        x = np.zeros(shape=(num_total_states, cwd_max_weight + 1))
        y = np.zeros(shape=(num_total_states, cwd_max_weight + 1))

        inS = 1
        outS = 3

        x[state, 0] = 1
        k = 0
        for t in np.arange(0, num_trellis_stages):

            for i in np.arange(0, num_total_states):
                y[i, :outS] = np.convolve(
                    x[dst_0[i], :inS], Wcoef0[i, :], mode="full"
                ) + np.convolve(x[dst_1[i], :inS], Wcoef1[i, :], mode="full")

            WBTBCCK1[t, i_worker, :] += y[state, :]

            # increment convolution input & output lengths
            inS = outS
            outS += 2
            k += 1

            if k == num_trellis_stages:
                # print(f"s: {i_state + i_worker}, state: {state}")
                print(f"when quitting, t={t}")
                break

            for i in np.arange(0, num_total_states):
                x[i, :outS] = np.convolve(
                    y[dst_0[i], :inS], Wcoef0[i, :], mode="full"
                ) + np.convolve(y[dst_1[i], :inS], Wcoef1[i, :], mode="full")

            WBTBCCK2[t, i_worker, :] += x[state, :]

            inS = outS
            outS += 2
            k += 1

            if k == num_trellis_stages:
                # print(f"s: {i_state + i_worker}, state: {basis[i_state + i_worker]}")
                print(f"when quitting, t={t}")
                break

BWs = np.zeros(shape=(num_trellis_stages, cwd_max_weight + 1))
oddIdxs = np.arange(0, num_trellis_stages, 2)
evenIdxs = np.arange(1, num_trellis_stages, 2)
h1 = np.arange(len(oddIdxs))
h2 = np.arange(len(evenIdxs))

BWs[oddIdxs, :] = np.sum(WBTBCCK1[h1, :, :], axis=1)
BWs[evenIdxs, :] = np.sum(WBTBCCK2[h2, :, :], axis=1)
print("WBTBCCK1 shape: ", WBTBCCK1.shape)
print("WBTBCCK2 shape: ", WBTBCCK2.shape)

np.set_printoptions(suppress=True)
print("BW shape: ", BWs.shape)
print(BWs[-1, :])

WBTBCC_name = f"DistanceSpec_V{nu}m{m}K{K}CRC{crc}"
WB_file_npy = f"output/{WBTBCC_name}.npy"
np.save(WB_file_npy, BWs)
