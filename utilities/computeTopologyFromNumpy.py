import ngl
import numpy as np
from hdff import *
import hdtopology as hdt

def main(args):

    data = None
    try:
        data = np.load(args.input)

    method = "RelaxedGabriel"
    max_neighbors = 500
    beta = 1.0

    ### provide recarray for data input ###
    data = np.concatenate((sample, np.matrix(f).T), axis=1).astype('f')
    names = ['X1', 'X2', 'X3', 'f']
    types = ['f4']*(d+1)
    data = data.view(dtype=list(zip(names,types)) ).view(np.recarray)
    print(data)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--histogram_dim', default=2, help='Precomputed datacube dimension', type=int)
    parser.add_argument('--graph_type', default=None, help="Specify neighborhood graph type", type=str)
    parser.add_argument('--input', default=None, help="Specify input filename", type=str)
    parser.add_argument('--output', help="Specify output filename", type=str, required=True)

    args = parser.parse_args(arguments)
    sys.exit(main(args))
