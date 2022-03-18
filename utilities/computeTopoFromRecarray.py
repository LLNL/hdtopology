import os
import argparse
import ngl
import numpy as np
from hdff import *
import hdtopology as hdt
import pandas as pd
###### test function ######

def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(description='Compute hdff file from recarray.')
    parser.add_argument('--inputfilename', type=str, help='Name of the input rearray numpy datafile.', required=True)
    parser.add_argument('--outputfilename', type=str, help='Name of output hdff datafile.', required=True)
    parser.add_argument('--method', type=str, help='method of the neighorhood graph.', default="RelaxedGabriel")
    parser.add_argument('--beta', type=float, help='beta parameter for gabriel graph.', default=1.0)
    parser.add_argument('--data_cube_dim', type=int, help='size of precomputed datacube.', default=2)

    parser.add_argument('--max_neighbors', type=int, help='method of the neighorhood graph.', default=500)
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()

    beta = args.beta

    ###### get domain and range from recarray ######
    data = np.load(args.inputfilename)
    print(data.dtype)
    domainNames = list(data.dtype.names[0:-1])
    print(domainNames)
    domain = data[domainNames] #.view('<f4')
    # print("domain:", domain.view('<f4').shape)
    # exit()
    # print(domain.shape)
    # domain = domain.view('<f4')
    # print(domain.shape, domain.dtype)
    domain = pd.DataFrame(domain).to_numpy()
    # exit()
    ### provide array of unint32 for the edges
    edges = ngl.getSymmetricNeighborGraph(args.method, domain, args.max_neighbors, beta)
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

    group = DataCollectionHandle(args.outputfilename)
    group.add(dataset)
    group.write()


if __name__ == '__main__':
    main()
