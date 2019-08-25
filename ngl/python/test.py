import ngl
import numpy as np
#import hdanalysis
from sys import argv

from hdanalysis.core import *

data = loadCSV(argv[1])

edges = ngl.getSymmetricNeighborGraph("RelaxedGabriel",data.asArray(),20,1.0)

print edges
