#ifndef EXTREMUM_GRAPH_H
#define EXTREMUM_GRAPH_H

#include <cstdint>
#include <vector>
#include <unordered_map>
#include "HDData.h"
#include "Neighborhood.h"
#include "Selectivity.h"
#include "JointDistributions.h"
#include <math.h>
#include "Flags.h"

// #define HD_FILE_FORMAT_H
// #ifdef HD_FILE_FORMAT_H
#include <DataBlockHandle.h>
// #endif

//! Forward declaration
class EdgeIterator;

class ExtremumGraphExt
{
public:

#ifndef SWIG
  struct Extremum {
    // Extremum(){}
    Extremum(uint32_t i=0, float ff=0, uint32_t p=0) : id(i), f(ff), persistence(10e34), parent(p) {}
    uint32_t id;
    float f;
    float persistence;
    uint32_t parent;
  };

  struct Saddle {
    Saddle(){}
    Saddle(uint32_t i, float ff, uint32_t u, uint32_t v) : id(i), f(ff), persistence(10e34) {neighbors[0]=u;neighbors[1]=v;cancellation=false;}
    uint32_t id;
    float f;
    uint32_t neighbors[2];
    float persistence;
    bool cancellation;
  };

  struct cmp {
    cmp(const HDData* d, const bool a) : data(d), ascending(a) {}

    //! Comparison operator between two vertices
    bool operator()(uint32_t i, uint32_t j) const;

    const HDData* data;
    const bool ascending;
  };
#endif

  enum ComputeMode {
    SEGMENTATION = 0,
    HISTOGRAM = 1,
    COMBINED = 2,
    NONE = 3
  };

  enum HistogramType {
    REGULAR = 0,
    REDUCED = 1,
    ENTROPY = 2,
    DTREE = 3
  };

  //! Constructor
  ExtremumGraphExt();

  //! Destructor
  ~ExtremumGraphExt() {}

  //! Construct an ExtremumGraph from a set of data, a neighborhood graph and the slopes
  void initialize(const HDData* data,  const Flags* flags, const Neighborhood* edges, bool ascending, uint32_t max_segments, const ComputeMode mode, uint32_t cube_dim = 2,
                  uint32_t resolution = 128, int32_t target_attr = -1,
                  std::vector<HistogramType> histogramTypes = std::vector<HistogramType> (1, REGULAR));

  void initialize(const HDData* data, const Flags* flags, EdgeIterator& edgeIter, bool ascending,
                  uint32_t max_segments, const ComputeMode mode, uint32_t cube_dim = 2,
                  uint32_t resolution = 128, int32_t target_attr = -1,
                  std::vector<HistogramType> histogramTypes = std::vector<HistogramType> (1, REGULAR));

  //! Return the segmentation for the given persistence
  uint32_t countForPersistence(float persistence);

  //! Return the segmentation for the given count of segments
  std::vector<std::vector<uint32_t> > segmentation(uint32_t count);


  //! Return the segment for the given ext
  std::vector<uint32_t> segment(uint32_t ext, uint32_t count, float threshold);

  //! Return the number of points for a given segment at the given
  //! segment count "above" the given threshold. The
  uint32_t segmentSize(uint32_t ext, uint32_t count, float threshold);

  //! Return the number of points for a given segment at the given
  //! count that are above the highest active saddle
  uint32_t coreSize(uint32_t ext, uint32_t count);

  //! Return the core segment for the given extremum
  std::vector<uint32_t> coreSegment(uint32_t ext, uint32_t count);

  //! Return the highest saddle for a given extremum
  int32_t highestSaddleForExtremum(uint32_t ext, uint32_t count);

  //! Return the number of segments
  uint32_t size() const {return mExtrema.size();}

  //! Return the list of persistences in descending order
  std::vector<float> persistences() const;

  //! Return the variations in descending order
  std::vector<float> variations() const;

  //! Returns a flat array of triples saddle, left, right
  std::vector<uint32_t> activeGraph(uint32_t count, float variation);

  float minimum() const {return mRange[0];}

  float maximum() const {return mRange[1];}

  float f(uint32_t i) const {return mFunction[i];}

  //! get extrema's information (using float is not ideal)
  std::vector<std::vector<float> > extrema() const;

  //! directly query saddle or extreme function value
  float criticalPointFunctionValue(uint32_t index);

  std::vector<float> criticalPointLocation(uint32_t index);


  // ! Query function for getting histogram with correct count, attrs and ext

  std::vector<uint32_t> getHist(uint32_t ext, uint32_t count, std::vector<std::string> attrs, bool func=false,
                                std::vector<std::string> dims=std::vector<std::string>(),
                                std::vector<std::vector<float> > ranges=std::vector<std::vector<float> >() );

  // ! Query function for getting histogram of the whole dataset

  std::vector<uint32_t> getHist(std::vector<std::string> attrs, bool func=false,
                                std::vector<std::string> dims=std::vector<std::string>(),
                                std::vector<std::vector<float> > ranges=std::vector<std::vector<float> >() );

  std::vector<uint32_t> getHist(std::string attr1, std::string attr2);

  //! get the global joint distribution
  JointDistributions& getJoint(){return mDistributions.back();}

  // ! Segmentation Function for Histogram only Mode
  std::vector<std::vector<uint32_t> > segmentHist(uint32_t count);

  // ! Other related funtion for Histogram only Mode
  uint32_t histogramSize(uint32_t ext, uint32_t count, float threshold);
  std::vector<uint32_t> histogram(uint32_t ext, uint32_t count, float threshold);

  // ! Get selected 1D based on query (This should return marginal distribution instead of probability)
  std::vector<uint32_t> getSelected1D(std::vector<std::string> dims, std::vector<std::vector<float> > ranges, int32_t ext, uint32_t count, int32_t targetIndex=-1);

  //! Interface for loading and writing to file
  bool save(HDFileFormat::DataBlockHandle &handle);
  //the flag indicate whether the dataset include the domain and range index for the function
  //if the flag is false, we assume the range is the last columne the previous n-1 are the domain
  bool load(HDFileFormat::DataBlockHandle &handle, bool isIncludeFunctionIndexInfo = false, uint32_t cube_dim = 2);
  // #endif
  //! serialization metaInfo
  void serialize(std::ostream &output);
  void deserialize(std::istream &input, bool isIncludeFunctionIndexInfo);

private:

  struct Triple {
    uint32_t saddle;
    float persistence;

    bool operator<(const Triple& u) const {return persistence > u.persistence;}
  };

  struct ExtremaCmp {
    ExtremaCmp(const std::vector<Extremum>& e) : extrema(e) {}

    bool operator()(uint32_t i,uint32_t j) const {
      return extrema[i].persistence < extrema[j].persistence;
    }

    const std::vector<Extremum>& extrema;
  };

  struct saddleCmp {
    bool operator()(const Saddle& u, const Saddle& v) const {return u.persistence < v.persistence;}
  };

  struct vertexCmp {
    vertexCmp(const std::vector<float>& f, const bool a) : function(f), ascending(a) {}

    //! Comparison operator between two vertices
    bool operator()(uint32_t i, uint32_t j) const;

    //! Comparison operator between two vertices
    bool operator()(uint32_t i, float f) const;

    //! Comparison operator between two vertices
    bool operator()(float f, uint32_t j) const;

    const std::vector<float> function;
    const bool ascending;

  };

  //! The array of function values ordered by mesh id
  std::vector<float> mFunction;

  //! The overall function range
  std::vector<float> mRange;

  //! The array of extrema ordered by descending persistence
  std::vector<Extremum> mExtrema;

  //! The array of saddles
  std::vector<Saddle> mSaddles;

  //! The array of steepest neighbors
  std::vector<uint32_t> mSteepest;

  //! The set of segments
  std::vector<std::vector<uint32_t> > mSegments;

  //! The currently buffered segmentation
  std::vector<std::vector<uint32_t> > mSegmentation;

  //! The segment count for the current segmentation
  uint32_t mSegCount;

  //! The index map from extrema mesh index to local index
  std::unordered_map<uint32_t, uint32_t> mIndexMap;

  //! Flag to indicate whether we used ascending or descending comparisons
  bool mAscending;

  //! The max dimension of the data cube stored
  uint32_t mCubeDim;

  //! Attr to indicate function
  // std::string mFuncAttr;
  uint32_t mFuncAttr;

  //! This stores jointdistributions of base partitions used to query histogram
  std::vector<JointDistributions> mDistributions;

  //! Selectivity Class
  Selectivity mSelectivity;

  //! This stores histogram of function value
  // std::vector<Histogram> mFunctionHist;
  std::vector<std::vector<uint32_t> > mFunctionHist;

  //! pointlocations for critical pts
  std::vector<std::vector<float> > mPointLocations;

  //! The currently buffered histograms
  // std::vector<Histogram> mBufferedFunctionHist;
  std::vector<std::vector<uint32_t> > mBufferedFunctionHist;

  //! Return the representative of the given extremum at the current count of extrema
  uint32_t rep(uint32_t v, uint32_t count=0);

  //! Return the local id of an active extremum
  int32_t activeExtremum(uint32_t ext, uint32_t count);

  //! Compute the initial segmentation.
  void computeSegmentation(const HDData* data,  const Flags* flags, const Neighborhood* edges, const ComputeMode mode);

  //! Base implementation to compute the initial segmentation.
  void computeSegmentation(const HDData* data, const Flags* flags,EdgeIterator& eIt, const ComputeMode mode);

  //! Construct the hierarchy of saddles and critical points
  void computeHierarchy();

  //! Sort all arrays by descending persistence
  void sort(const HDData* data);

  //! Temporary function to store pts location for updated critical pts
  void storeLocations(const HDData* data);

  //! Compute the current persistence of a saddle
  float computePersistence(uint32_t saddle);

  //! Compute the individual segments
  void computeSegments(const HDData* data);

  //! Compute the histogram information
  void computeHistograms(const HDData* data, uint32_t cube_dim, uint32_t resolution,
                          std::vector<HistogramType> histogramTypes, int32_t target_attr);

  //! Simplify the hierarchy to the given number of extrema. Note that this operation
  //! is *destructive*. Also this
  void simplify(uint32_t count);

  //! Return the set of all active saddles
  std::vector<uint32_t> activeSaddles(uint32_t count, float saturated);

  //! serialization buffer
  std::vector<char> mSerializationBuffer;

};


#endif
