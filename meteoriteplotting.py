# Plotting support for determining equation that I will need to graph
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Create a new csv reader
data = pd.read_csv("cleanlandings.csv")
print(data)

# Create panda dataframe
dataframe = pd.DataFrame(data, columns=["id", "rectype", "recmass", "recfalltype", "reclat", "reclong"])

# Plot the data
sns.jointplot(x="reclong", y="reclat", data=dataframe)
plt.show()