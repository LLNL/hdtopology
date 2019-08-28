%{
	#include "MorseComplex.h"
%}

%include "MorseComplex.h"


%pythoncode %{

def construct_complex(complex,f):

  morse = MorseComplex()
  morse.setSegmentation(complex.hierarchy)

  reps = [x.representative for x in complex.hierarchy]
  #print reps
  morse.setIndexMap(reps)

  # Collect all saddles that are part of cancellations
  saddles = dict()
  for seg in complex.hierarchy:
      #print seg.representative,seg.parent,seg.persistence
      morse.addCancellation(seg.representative,seg.parent,seg.persistence)
      #print seg.representative.__class__,f[seg.representative][-1].__class__
      morse.setNodeInfo(seg.representative, f[seg.representative][-1])

      if seg.persistence < 10e33:
          saddles[seg.saddle] = seg.persistence

  order = range(0,len(complex.hierarchy))
  morse.setOrder(order)


  # Now finally add all the saddles triples
  for seg in complex.hierarchy: # For all segments
      for neigh in seg.neighbors: #For all neighbors
          if seg.representative < neigh: # Only output each edge once
              s = seg.neighbors[neigh] # THe corresponding saddles
              if s in saddles: # Is this a saddle that cancels
                  morse.addSaddlePair(s,f[s][-1],seg.representative,neigh,saddles[s])
              else:
                  morse.addSaddlePair(s,f[s][-1],seg.representative,neigh,10e34)


  morse.finalizeConstruction()

  return morse

%}
