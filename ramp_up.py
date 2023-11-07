import logging

import numpy as np
from mpi4py import MPI

logging.basicConfig(format='%(name)s: %(message)s')

# Initialize MPI communication
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

logger = logging.getLogger(str(comm.Get_rank()))
logger.warning(f"MPI: rank {rank} / {size}")

# Load data
data = np.loadtxt(r'cluster_data.csv', delimiter=",", skiprows=1)
data = data[:, 1:]
# Divide data evenly among processes
local_data = np.array_split(data, size)[rank]

# my_local_data = comm.scatter(local_data, root=0)
#logger.warning(local_data)
# print(local_data.shape)


# Calculate local preliminary centroids and number of assigned points
local_centroid = np.mean(local_data, axis=0)
local_n_points = local_data.shape[0]

# Collect preliminary centroids and number of assigned points at coordinator
centroids = comm.gather(local_centroid, root=0)
n_points = comm.gather(local_n_points, root=0)

logger.warning(centroids)
logger.warning(n_points)

# Compute final centroids at coordinator
if rank == 0:
    total_n_points = sum(n_points)
    weighted_centroids = [c * n for c, n in zip(centroids, n_points)]
    global_centroid = sum(weighted_centroids) / total_n_points
    print("Global centroid:", global_centroid)
