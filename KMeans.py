
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
import random as rd
import matplotlib.pyplot as plt


def get_random_cluster_center():
    new_mu = []
    for x in range(9):
        new_mu.append(rd.random() * 15)
    return np.array(new_mu)

def check_termination(old_mus, mus):
    allclose = np.allclose(old_mus, mus)
    return allclose
            

# begin
ds = pd.read_csv('bc.txt', header=None, usecols=[x for x in range(1,10)], sep=',')

kvalues = [2,3,4,5,6,7,8]
potential = [0.0 for x in range(len(kvalues))]

for K in range(len(kvalues)):
    
    k = kvalues[K]

    # generate random cluster locations
    mus = []
    for i in range(k):
        while(True):
            new_mu = get_random_cluster_center()
            equal = False
            for mu in mus:
                if np.array_equal(mu, new_mu):
                    equal = True
            if not equal:
                mus.append(new_mu)
                break

    old_mus = mus[:]

    while(True):
        # classify step
        c = [0 for i in range(len(ds))]
        sizes = [[] for x in range(k)]
        for index, row in ds.iterrows():
            point = row.values
            min_sum = 1000000.00
            for i in range(k):
                sum = np.sum(np.power(np.subtract(mus[i], point), 2))
                if sum < min_sum:
                    min_sum = sum
                    c[index] = i
            sizes[c[index]].append(index)
 
        # handle empty clusters
        for i in range(len(sizes)):
            if not sizes[i]:
                # find largest cluster
                max_size = 0
                max_size_index = 0
                for j in range(len(sizes)):
                    if len(sizes[j]) > max_size:
                        max_size = len(sizes[j])
                        max_size_index = j

                # split randomly
                new_list = []
                for m in range(int(len(sizes[max_size_index])/2),len(sizes[max_size_index])):
                    c[sizes[max_size_index][m]] = i
                    new_list.append(sizes[max_size_index][m])

                for l in new_list:
                    sizes[i].append(l)
                    sizes[max_size_index].remove(l)

        # recenter
        old_mus = mus[:]
        for i in range(len(sizes)):
            new_mu = np.array([0.0 for x in range(9)])
            for index in sizes[i]:
                point = ds.iloc[index].values
                new_mu = np.add(new_mu, point)
            new_mu = np.divide(new_mu, float(len(sizes[i])))
            mus[i] = new_mu

        if check_termination(old_mus, mus):
            # compute the potential function
            for p in range(len(c)):
                term = np.sum(np.power(np.subtract(mus[c[p]], ds.iloc[p].values), 2))
                potential[K] += term
            break

plt.plot(kvalues, potential)
plt.savefig('graph.png')

