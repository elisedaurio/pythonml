# Plotting support for determining equation that I will need to graph
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Create a new csv reader
data = pd.read_csv("cleanlandings.csv")

# Create panda dataframe
dataframe = pd.DataFrame(data, columns=["id", "rectype", "recmass", "recfalltype", "reclat", "reclong"])

dataframe.plot.hexbin(x="reclong", y="reclat", C="recmass", bins = "log", reduce_C_function=np.max, gridsize=50)
plt.show()