from hdff import *
import hdtopology as hdt

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
attrs = eg.getJoint().getAttr()
print("Attrs:", attrs)
hist = eg.getHist(attrs[:2])
print("Histogram Bin Value Range:", hist.min(), hist.max())
