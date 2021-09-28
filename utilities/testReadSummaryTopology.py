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

print(eg.f(0), eg.minimum(), eg.maximum())
print(eg.extrema())
ind = eg.extrema()[1][2]
print(eg.segment(ind, 2, eg.minimum()))
print(eg.segmentation(1))
print(eg.coreSegment(ind, 2))
