import ngl
import numpy as np
from hdff import *
import hdtopology as hdt

n = 1000
d = 3
sample = np.random.uniform(-1.0,1.0,n*d).astype('f')
sample = sample.reshape(n, d)

###### test function ######
def ackley(domain, d=3):
    # print "domain", domain.shape
    Sum = np.zeros(domain.shape[0], dtype=float)
    for i in range(d-1):
        theta1 = 6*domain[:,i] - 3
        theta2 = 6*domain[:,i+1] - 3
        # sum -= exp(-0.2) * sqrt(pow(theta1, 2) + pow(theta2, 2)) + 3 * (cos(2 * theta1) + sin(2 * theta2));
        # print(theta1.shape, Sum)
        Sum = Sum - np.exp(-0.2) * np.sqrt(np.power(theta1, 2) + np.power(theta2, 2)) + 3 * (np.cos(2 * theta1) + np.sin(2 * theta2));

    Sum = np.squeeze(np.array(Sum.T))
    # print(Sum.shape)
    return Sum

f = ackley(sample)

method = "RelaxedGabriel"
max_neighbors = 500
beta = 1.0

### provide recarray for data input ###
data = np.concatenate((sample, np.matrix(f).T), axis=1).astype('f')
names = ['X1', 'X2', 'X3', 'f']
types = ['<f4']*(d+1)
data = data.view(dtype=list(zip(names,types)) ).view(np.recarray)
print(data.dtype)
### provide array of unint32 for the edges
edges = ngl.getSymmetricNeighborGraph(method, sample, max_neighbors,beta)
print(edges, type(edges), edges.dtype)
### compute topology
eg = hdt.ExtremumGraphExt()
flag_array = np.array([0],dtype=np.uint8)
eg.initialize(data, flag_array, edges, True ,10, 1)

mc = DataBlockHandle()
mc.idString("TDA");
eg.save(mc)
dataset = DatasetHandle()
dataset.add(mc)

group = DataCollectionHandle("summaryTopologyTest.hdff")
group.add(dataset)
group.write()
