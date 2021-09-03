import os
import argparse
import ngl
import numpy as np
from hdff import *
import hdtopology as hdt


###### test function ######

def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(description='Compute hdff file from recarray.')
    parser.add_argument('--inputfilename', type=str, help='Name of the input rearray numpy datafile.', required=True)
    parser.add_argument('--outputfilename', type=str, help='Name of output hdff datafile.', required=True)
    parser.add_argument('--method', type=str, help='method of the neighorhood graph.', default="RelaxedGabriel")
    parser.add_argument('--max_neighbors', type=int, help='method of the neighorhood graph.', default=500)
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()

    beta = 1.0

    ### provide recarray for data input ###
    data = np.load(args.inputfilename)
    domainNames = data.names[0:-1]
    print(domainNames)
    domain = data[domainNames].copy().view('<f8')
    print("domain:", domain.shape)
    ### provide array of unint32 for the edges
    edges = ngl.getSymmetricNeighborGraph(args.method, domain, args.max_neighbors, beta)
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

    group = DataCollectionHandle(args.outputfilename)
    group.add(dataset)
    group.write()


if __name__ == '__main__':
    main()
