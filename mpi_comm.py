#!/usr/bin/env python3

"""

mpi_comm
========

This module contains help functions for MPI communication.

"""

import math
import numpy as np
from mpi4py import MPI
from itertools import islice
import time


# MPI variables
comm = MPI.COMM_WORLD
rank = comm.rank
ranks = comm.size


def dict_chunks_from_one_MPI_rank(data, chunk_maxsize=1*10**6, root=0):
    """
    Divide up date in chunks for one MPI rank.

    Yields chunks of data.
    Each chunk will contain a maximum number of elements,
    which is determined by the user.
    The other MPI ranks yields the same number of chunks,
    but each such chunk is equal to None.

    Parameters
    ----------
    data : dict
    chunk_maxsize : int
    root : int

    """
    if rank == root:
        it = iter(data)
        n_chunks = math.ceil(len(data)/chunk_maxsize)
    else:
        n_chunks = None
    n_chunks = comm.bcast(n_chunks, root=root)
    for _ in range(n_chunks):
        if rank == root:
            yield {k:data[k] for k in islice(it, chunk_maxsize)}
        else:
            yield None


def allgather_dict(data, total, chunk_maxsize=1*10**6):
    """
    Distribute data from all ranks to all ranks into variable total.

    The function performs "Allgather".
    However, since Allgather requires the same amount of data
    for all MPI ranks, it's done through simpler communications.

    Parameters
    ----------
    data : dict
        Contains different information for each MPI rank.
        Unique keys for each rank, i.e.
        a key for rank r does not exist as a key in data
        for any other rank other than rank r.
        Neither does it exist in the variable total.
    total : dict
        Will be updated with data from all MPI ranks.
    chunk_maxsize : int
        The maximum number of dictionary elements to send at once.

    """
    # Measure time for constructing H in matrix form
    t0 = time.time()
    # Number of elements for each rank.
    n_ps_new = np.zeros(ranks, dtype=np.int)
    for r in range(ranks):
        n_ps_new[r] = comm.bcast(len(data), root=r)
    # Determine here if we can use a simple Allgather or need
    # to send the data in chunks.
    if max(n_ps_new) <= chunk_maxsize:
        if rank == 0: print('Allgather everything at once...')
        for r in range(ranks):
            total.update(comm.bcast(data, root=r))
    else:
        if rank == 0: print('Allgather chunks...')
        # MPI do not allow to messages bigger than about 2 GB.
        # Therefore we send the data in chunks.
        #print('rank' + str(rank) +', h_big_new = ', h_big_new)
        for r in range(ranks):
            # Data in rank r is broadcasted in chunks to all the other ranks.
            for chunk in dict_chunks_from_one_MPI_rank(data, chunk_maxsize, r):
                #print('rank' + str(rank) + ': ', chunk)
                total.update(comm.bcast(chunk, root=r))
    if rank == 0:
        print("time(Allgather H_dict) = {:.5f} seconds.".format(time.time()-t0))
