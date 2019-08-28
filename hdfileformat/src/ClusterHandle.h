#ifndef CLUSTER_HANDLE_H
#define CLUSTER_HANDLE_H

#include <vector>
#include <string>
#include <algorithm>

#include "DataBlockHandle.h"
#include "SubspaceHandle.h"
#include "HierarchyHandle.h"

#include "ExplicitHierarchy.h"
// #include "ImplicitHierarchy.h"

/*
* Handle for various cluster results
* Author: Shusen Liu  Date: Oct 1, 2014
* Parameter string pattern "Param1 = value1, Param2 = value2"
*/

namespace HDFileFormat{

enum ClusteringResultType{
  ClusterType_FlatCluster,
  ClusterType_FlatCluster_Subspace,
  ClusterType_HierarchicalCluster,
  ClusterType_UnKnown
};

class ClusterHandle : public DataBlockHandle
{
public:

  //! The default name of the clustering result
  static const std::string sDefaultClusterName;

  //! Constructor
  explicit ClusterHandle(HandleType t=H_CLUSTER);

  //! Constructor
  ClusterHandle(const char *filename, HandleType t=H_CLUSTER);

  //! Copy constructor
  ClusterHandle(const ClusterHandle& handle);

  //! Destructor
  virtual ~ClusterHandle();

  //! Assignement operator
  ClusterHandle& operator=(const ClusterHandle& handle);

  //! Clone this handle
  virtual FileHandle* clone() const {return new ClusterHandle(*this);}

  //! add children
  virtual FileHandle& add(const FileHandle &handle);

  /// clustering result specific acessor ///
  void setLabel(uint32_t *label, int pointCount);

  //! Get clustering result type
  ClusteringResultType GetClusteringResultType();

  //! Get cluster label, index corresponding to point index
  //std::vector<int>& GetFlatClusterLabel();

  //! Get point indices for each label, label[i] contain all the index belong to label i
  //std::vector<std::vector<int> >& GetFlatClusterLabelPointIndex();

  //virtual int readData(int* label);

  //! Get the file size
  //virtual uint32_t size() const {return mSampleCount*sizeof(int);}

  //! Get the number of samples
  uint32_t sampleCount() const {return mSampleCount;}

  //! Set the cluster method parameter string
  void setParameterString(std::string params)
  {mClusteringParameterString = params;}

  //! Get the cluster method parameter string
  std::string& parameterString(){return mClusteringParameterString;}

  //! Read into common object that could shared by other applications
  void readHierarchy(HDFileFormat::ExplicitHierarchy& hierarchy);

  //! Read into common object that could shared by other applications
  // void readHierarchy(HDFileFormat::ImplicitHierarchy& hierarchy);

protected:

  //! Parse the information from the xml tree
  virtual int parseXMLInternal(const XMLNode& node);

  //! Add the local attribute to the node
  virtual int attachXMLInternal(XMLNode& node) const;

protected:
  std::string mClusteringParameterString;
  // TODO
  //ClusteringResultType mClusterResultType;
  //std::vector<int> mFlatLabel;
  //std::vector<std::vector<int> > mFlatLabelPointIndex;

};

}

#endif
