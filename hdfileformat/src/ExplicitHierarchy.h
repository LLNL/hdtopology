#ifndef EXPLICIT_HIERARCHY_H
#define EXPLICIT_HIERARCHY_H

#include <vector>
#include <queue>
#include <string>
#include <stddef.h>

namespace HDFileFormat{

/*
 * The explicitHierarchy store the hierarchy as tree in memory
 * Author: Shusen Liu  Date: Oct 14, 2014
*/

struct hNode
{
  hNode(hNode *p = NULL, bool leaf = false)
  {
    parent = p;
    isLeaf = leaf;
  }
  
  std::vector<hNode*> GetAllChildrenLeafNode()
  {
    std::vector<hNode*> leafNodes;
    std::queue<hNode*> nodeQueue;
    nodeQueue.push(this);
    
    //BFS of the tree
    while(!nodeQueue.empty())
    {
      hNode *node = nodeQueue.front();
      nodeQueue.pop();
      
      if(!node->isLeaf)
      {      
        for(size_t i=0; i<node->childrenList.size(); i++)
          nodeQueue.push(node->childrenList[i]);
      }
      else //node is leaf
      {
        leafNodes.push_back(node);
      }
    }
    return leafNodes; 
  }
  
  std::vector<hNode*> childrenList;
  hNode *parent;
  bool isLeaf;
  std::vector<int> leafLabels; // size = 1 ... n  
};

class ExplicitHierarchy
{
public:
  ExplicitHierarchy();
  ~ExplicitHierarchy();

  //! access node root
  hNode *root();
    
  //! access the raw buffer
  void setRawBuffer(void *buffer, int bufferSize);
  void getRawBuffer(void *buffer, int &bufferSize);
  
  //! access the data for writing usage for HierarchyHandle
  // only the hierarchyHandle should be able to access these information
  
protected:
  void parseHierarchy();
  
  std::string mHierarchy;

  hNode *mRoot;
  
};


}

#endif
