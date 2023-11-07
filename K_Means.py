import logging

import numpy as np
from mpi4py import MPI
import pandas as pd

logging.basicConfig(format='Rank %(name)s: %(message)s')

# Initialize MPI communication
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

logger = logging.getLogger(str(comm.Get_rank()))
logger.warning(f"MPI: rank {rank} / {size}")

N = 400
M = 2
K = 4

centroids_matrix = np.zeros([K, M])

my_data = None

# Load data
if rank == 0:
    data = pd.read_csv(r'cluster_data.csv', header=0, index_col=0)
    data = data.values
    my_data = np.array_split(data, size)
    initial_ids = np.random.choice(len(data), K, replace=False)
    centroids_matrix = data[initial_ids]

# In summary, scatter distributes data from a root process to individual processes,
# while bcast shares the same data with all processes in the communicator.

my_data = comm.scatter(my_data, root=0)
centroids_matrix = comm.bcast(centroids_matrix, root=0)
distance_matrix = np.zeros([len(my_data), K])
cluster_assignment = np.zeros(len(my_data))

def l2_dist(point, cluster):
    return np.sqrt(sum((point - cluster)**2))

for itr in range(10):
    for row in range(len(my_data)):
        for cluster_id in range(K):
            distance_matrix[row, cluster_id] = l2_dist(my_data[row], centroids_matrix[cluster_id])
    cluster_assignment = np.argmin(distance_matrix, axis=1)

    sub_centroids = np.zeros([K, M])
    for cluster_id in range(K):
        ids = []
        for i in range(len(my_data)):
            if cluster_assignment[i] == cluster_id:
                ids.append(i)
        if len(ids) == 0:
            sub_centroids[cluster_id] = np.array([0, 0])
        else:
            points = my_data[ids]
            sub_centroids[cluster_id] = len(ids) * np.mean(points, axis=0)

    sub_centroids = comm.gather(sub_centroids, root=0)
    if rank == 0:
        sub_centroids = np.array(sub_centroids)
        centroids_matrix = sub_centroids.sum(axis=0)
        centroids_matrix /= N
    centroids_matrix = comm.bcast(centroids_matrix, root=0)

if rank == 0:
    logger.warning(f"Final centroids: ")
    logger.warning(centroids_matrix)

