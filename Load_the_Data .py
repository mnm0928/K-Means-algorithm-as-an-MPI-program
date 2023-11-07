import matplotlib.pyplot as plt
import pandas as pd

# Load data
data = pd.read_csv(r'cluster_data.csv', header=0, index_col=0)

# Plot data
plt.scatter(data["x"], data["y"])
plt.title("Cluster Data")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

