import numpy as np
from hdff import *
import hdtopology as hdt
import numpy as np
from sys import argv,exit
import matplotlib.pyplot as plt


if len(argv) < 3:
    print("Usage: %s image.raw dim-x dim-y" % argv[0])
    exit(0)
    
# For demonstration purposes we will read in a raw float file
data = np.fromfile(argv[1],dtype=np.float32)
dimx = int(argv[2])
dimy = int(argv[3])

#data = data[0:9]


img = data.reshape([dimy,dimx])

# Now create a 2D mesh to put the function on 
X, Y = np.meshgrid(range(0,dimx), range(0,dimy))

# Make a rec-array out of it for hdtopology
data = np.stack((X.flatten(),Y.flatten(),data), axis=1).astype('f')
names = ['X', 'Y', 'f']
types = ['f4']*3
data = data.view(dtype=list(zip(names,types)) ).view(np.recarray)

edges = []
for j in range(0,dimy-1):
    for i in range(0,dimx-1):
        edges.append([j*dimx + i, j*dimx + i + 1])
        edges.append([j*dimx + i, (j+1)*dimx + i])
        
    edges.append([j*dimx + dimx - 1, (j+1)*dimx + dimx - 1])

for i in range(0,dimx-1):
    edges.append([(dimy-1)*dimx + i, (dimy-1)*dimx + i + 1])

# Now we make sure that the edges are symmetric
edges2 = [[e[1],e[0]] for e in edges]
edges += edges2

edges = np.array(edges,dtype=np.uint32)    



eg = hdt.ExtremumGraphExt()
flag_array = np.array([0],dtype=np.uint8)
eg.initialize(data, flag_array, edges, True ,10, 0)

fig = plt.figure(figsize=(1, 3))

fig.add_subplot(1,3, 1)
# Show the orinal image
plt.imshow(img)

# Now we make a segmented image from the hierarchy

# Create an empty image
discrete = img.copy().flatten()

# Get k number of segments
segs = eg.segmentation(3)

for k,s in enumerate(segs):
    for i in s:
        discrete[i] = k 

fig.add_subplot(1,3, 2)
plt.imshow(discrete.reshape([dimy,dimx]))


# Get k number of segments
segs = eg.segmentation(6)

for k,s in enumerate(segs):
    for i in s:
        discrete[i] = k 

fig.add_subplot(1,3, 3)
plt.imshow(discrete.reshape([dimy,dimx]))


plt.show()



