import os
import sys
import argparse
import numpy as np

import ngl
from hdff import *
import hdtopology as hdt

def main(args):

    data = None
    try:
        dataset = np.load(args.input)
    except:
        print('Can not load:', args.input)
        exit()

    method = "RelaxedGabriel"
    max_neighbors = args.max_neighbors
    beta = 1.0

    sample = dataset[:, :-1].astype('float32')
    print('samples:', sample.shape, sample.dtype)
    f = dataset[:, -1]
    print('f:', f.shape, f.dtype)

    data = np.concatenate((sample, np.matrix(f).T), axis=1).astype('f')

    ### provide recarray for data input ###
    names = ["D"+str(i) for i in range(sample.shape[1])]+['f']
    types = ['f4']*len(names)
    print("names:", names)

    data = data.view(dtype=list(zip(names,types)) ).view(np.recarray)
    print('data shape:', data.shape)

    ### provide array of unint32 for the edges
    edges = ngl.getSymmetricNeighborGraph(method, sample, max_neighbors,beta)
    print(edges, type(edges), edges.dtype)

    ### compute topology
    eg = hdt.ExtremumGraphExt()
    flag_array = np.array([0],dtype=np.uint8)
    mode = 0
    if args.data_cube_dim>1:
        mode = 1
    eg.initialize(data, flag_array, edges, True ,10, mode, args.data_cube_dim)

    mc = DataBlockHandle()
    mc.idString("TDA");
    eg.save(mc)
    dataset = DatasetHandle()
    dataset.add(mc)

    group = DataCollectionHandle(args.output)
    group.add(dataset)
    group.write()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_cube_dim', default=2, help='Precomputed datacube dimension', type=int)
    parser.add_argument('--max_neighbors', default=300, help='max neighborhood size', type=int)

    parser.add_argument('--graph_type', default=None, help="Specify neighborhood graph type", type=str)
    parser.add_argument('--input', default=None, help="Specify input filename", type=str)
    parser.add_argument('--output', help="Specify output filename", type=str, required=True)


    args = parser.parse_args()
    sys.exit(main(args))
