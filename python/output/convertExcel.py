import numpy as np
import pandas as pd

# 1. Load the .npy file
filename = "k51n126v8_dist_spectrum.npy"
data = np.load("fold/" + filename)

# 2. Convert to a Pandas DataFrame
# If your data is 1D or 2D, this works perfectly.
df = pd.DataFrame(data)

# 3. Save to Excel
df.to_excel("k51n126v8_dist_spectrum.xlsx", index=False)

print("Conversion complete!")
