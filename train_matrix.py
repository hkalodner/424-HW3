#!/usr/bin/env python3

import numpy as np
from scipy.sparse import csr_matrix

train_data = np.load("txTripletsCounts.npy")
dim = np.max(train_data) + 1
matrix = csr_matrix((dim, dim), dtype = bool)

matrix[train_data[:, 0], train_data[:, 1]] = True
