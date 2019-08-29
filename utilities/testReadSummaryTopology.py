from hdff import *
from hdtopology import hdt

filename = "summaryTopologyTest.hdff"
collection = DataCollectionHandle()
collection.attach(filename)
dataset = collection.dataset(0)
# load EG
eg = hdt.ExtremumGraphExt()
handle = dataset.getDataBlock(0)
isIncludeFunctionIndexInfo = False
cube_dim = 2

eg.load(handle, isIncludeFunctionIndexInfo, cube_dim)

##### test query ######
