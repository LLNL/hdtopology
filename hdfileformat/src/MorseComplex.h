/*
 * MorseComplex.h
 *
 *  Created on: Feb 12, 2015
 *      Author: bremer5
 */

#ifndef MORSECOMPLEX_H
#define MORSECOMPLEX_H

#include "HierarchicalSegmentation.h"
#include "TopoGraph.h"
#include "DistributionHandle.h"

namespace HDFileFormat {


//! A simple Morse complex on top of a hierarchical segmentation
class MorseComplex : public HierarchicalSegmentation
{
public:

  //! Constructor
  MorseComplex();

  //! Destructor
  virtual ~MorseComplex() {}

  //! Initialize the segmentation from a segmentation handle
  int initialize(SegmentationHandle& handle);

  //! Construct a segmentation handle
  virtual SegmentationHandle makeHandle();

  //! Add saddle pair
  void addSaddlePair(uint32_t saddle, float f, uint32_t a, uint32_t b, float p);

  //! Finalize the saddles by sorting them
  void finalizeConstruction();

  //! Set the node info corresponding to the representatives/extrema
  void setNodeInfo(uint32_t global_index, float function);// HDFileFormat::NodeType type=MAXIMUM);

  //! Return the cancellation tree at the given persistence as a graph
  void cancellationTree(HDFileFormat::TopoGraph& tree,float persistence) const;

  //! Return the topological spine at the given persistence as a graph
  void topologicalSpine(HDFileFormat::TopoGraph& tree,float persistence, float ridgeness) const;

  //! Return the persistence value for a given node number
  float getPersistenceByNodeNumber(int nodeNum);

  //! Return the max persistence in the persistence plot
  float getPersistenceMax(){return mMaxPersistence;}

  //! Return persistence vs. extrema
  std::vector<std::pair<float, int> >& getPersistenceVersusExtrema(){return mPersistenceVersusExtrema;}

  //! Return persistence vs. num. arcs
  std::vector<std::pair<float, int> >& getPersistenceVersusArcs(){return mPersistenceVersusArcs;}

  void writeDot(const char* filename) const;
//protected:

  //! A saddle pair
  class SaddlePair {
  public:
    uint32_t saddle; //! The index of the saddle
    float function; //Function value of the saddle
    uint32_t neigh[2]; //! The two neighboring extrema
    float parameter; //! The simplification parameter

    friend std::istream& operator>>(std::istream &input, MorseComplex::SaddlePair &info);
    friend std::ostream& operator<<(std::ostream &output, MorseComplex::SaddlePair &info);
  };
  //need to be friend of the outter class as well
  friend std::istream& operator>>(std::istream &input, MorseComplex::SaddlePair &info);
  friend std::ostream& operator<<(std::ostream &output, MorseComplex::SaddlePair &info);

  //! The function value and type of a node
  class NodeInfo
  {
  public:

    float function;
    NodeType type;

    friend std::istream& operator>>(std::istream &input, MorseComplex::NodeInfo &info);
    friend std::ostream& operator<<(std::ostream &output, MorseComplex::NodeInfo &info);

  };
  //need to be friend of the outter class as well
  friend std::istream& operator>>(std::istream &input, MorseComplex::NodeInfo &info);
  friend std::ostream& operator<<(std::ostream &output, MorseComplex::NodeInfo &info);

  //! Parameter based comparison
  static bool pairCmp(const SaddlePair& a, const SaddlePair& b);

  //! The list of saddles sorted by decreasing parameter
  std::vector<SaddlePair> mSaddles;

  //! The list of function values and node types
  std::vector<NodeInfo> mNodes;

protected:
  void mUpdatePersistencePlotData();

  //! for persistence plot
  std::vector<std::pair<float, int> > mPersistenceVersusExtrema;
  std::vector<std::pair<float, int> > mPersistenceVersusArcs;

  //! maxPersistence in the plot
  float mMaxPersistence;
};


#ifndef SWIG
std::istream& operator>>(std::istream &input, MorseComplex::SaddlePair &info);
std::ostream& operator<<(std::ostream &output, MorseComplex::SaddlePair &info);

std::istream& operator>>(std::istream &input, MorseComplex::NodeInfo &info);
std::ostream& operator<<(std::ostream &output, MorseComplex::NodeInfo &info);

#endif

}

#endif
