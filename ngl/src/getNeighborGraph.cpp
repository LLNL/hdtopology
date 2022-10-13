#include <set>
#include <map>
#include <cstdio>
#include <cstring>
#include <algorithm>

#include "getNeighborGraph.h"



#define METHOD_COUNT 9
const char* method_names[METHOD_COUNT] = {
  
  "Ann",

  "Gabriel",
  "RelativeNeighbor",
  "BSkeleton",
  "Diamond",

  "RelaxedGabriel",
  "RelaxedRelativeNeighbor",
  "RelaxedBSkeleton",
  "RelaxedDiamond",
};

using namespace ngl;

typedef void (*fn)(NGLPointSet<float> &points, IndexType **indices, int &numEdges, NGLParams<float> params);

//typedef vector< tuple<ngl::IndexType, ngl::IndexType, float> > iif_tuple;
typedef std::vector< std::tuple<ngl::IndexType, ngl::IndexType, float>> result_tuple;

static std::map<std::string, fn> gMethods;

void init_methods()
{
  gMethods["Ann"] = getKNNGraph;
  gMethods["RelativeNeighbor"] = getRelativeNeighborGraph;
  gMethods["Gabriel"] = getGabrielGraph;
  gMethods["BSkeleton"] = getBSkeleton;
  gMethods["Diamond"] = getDiamondGraph;
  gMethods["RelaxedRelativeNeighbor"] = getRelaxedRelativeNeighborGraph;
  gMethods["RelaxedGabriel"] = getRelaxedGabrielGraph;
  gMethods["RelaxedBSkeleton"] = getRelaxedBSkeleton;
  gMethods["RelaxedDiamond"] = getRelaxedDiamondGraph;
}  

class Edge
{
public: 
  
  Edge(ngl::IndexType u, ngl::IndexType v) {mE[0] = std::min(u,v), mE[1]=std::max(u,v);}
  
  Edge(const Edge& e) {mE[0] = e[0]; mE[1] = e[1];}

  ngl::IndexType& operator[](int i) {return mE[i];}
  const ngl::IndexType& operator[](int i) const {return mE[i];}

  bool operator<(const Edge& e) const {
    return ((mE[0] < e[0]) || ((mE[0] == e[0]) && (mE[1] < e[1])));
  }

  ngl::IndexType mE[2];
};

void getNeighborGraph(const char* method, 
                      ANNPointSet<float>* points,
                      int kmax, 
                      float param,
                      IndexType** edges, int* numEdges)
{
  init_methods();
  
  int i;
  
  std::map<std::string, fn>::iterator mIt;

  mIt = gMethods.find(method);
  
  if (mIt == gMethods.end()) {
    fprintf(stderr,"Did not recognize method name %s\n",method);
    assert(false);
  }    
  
  
  NGLParams<float> params;
  
  params.iparam0 = kmax;
  params.param1 = param;
  
  mIt->second(*points, edges, *numEdges, params);

  return;
}

void getNeighborGraph_beta(ANNPointSet<float>* points,
                      int kmax, 
                      float param,
                      IndexType** edges, int* numEdges, float** betas)
{
  
  int i;
  
  //std::map<std::string, fn>::iterator mIt;

  //mIt = gMethods.find(method);
  
  //if (mIt == gMethods.end()) {
   // fprintf(stderr,"Did not recognize method name %s\n",method);
    //assert(false);
  //}    
  
  
  NGLParams<float> params;
  
  params.iparam0 = kmax;
  params.param1 = param;
  
  getBSpectrum(*points, edges, betas, *numEdges, params);

  return;
}


void getSymmetricNeighborGraph(const char* method, 
                               ANNPointSet<float>* points,
                               int kmax, 
                               float param,
                               std::vector<ngl::IndexType>* edges)
{
  IndexType* asym;
  int num_asym;
  std::map<ngl::IndexType,int> index_map;
  std::map<ngl::IndexType,int>::iterator mIt;
  std::set<Edge> open;
  if (strcmp(method, "BSpectrum") == 0) {
  float* asym_beta;
  getNeighborGraph_beta(points,kmax,param,&asym,&num_asym, &asym_beta);

  edges->reserve(2*num_asym);
  
  for (int i=0;i<2*num_asym;i+=2) {
    if (open.find(Edge(asym[i],asym[i+1])) == open.end()) {
      edges->push_back(asym[i]);
      edges->push_back(asym[i+1]);
      edges->push_back(asym[i+1]);
      edges->push_back(asym[i]);
      if (i==0){
      fprintf(stderr,"%u %u %f\n",asym[i],asym[i+1],asym_beta[i]);
      fprintf(stderr,"%u %u %f\n",asym[i+1],asym[i],asym_beta[i]);
      } else if ( i < num_asym){
      fprintf(stderr,"%u %u %f\n",asym[i],asym[i+1],asym_beta[i-1]);
      fprintf(stderr,"%u %u %f\n",asym[i+1],asym[i],asym_beta[i-1]);
  
      } 

      //fprintf(stderr,"%u %u %f\n",asym[i],asym[i+1],asym_beta[i-1]);
      //fprintf(stderr,"%u %u %f\n",asym[i+1],asym[i],asym_beta[i-1]);
      open.insert(Edge(asym[i],asym[i+1]));
    }
    else 
      //fprintf(stderr,"about to seg\n"); 
      open.erase(Edge(asym[i],asym[i+1]));   
  }

  delete[] asym;
  delete[] asym_beta; 
  return;
  } else {
  
  getNeighborGraph(method,points,kmax,param,&asym,&num_asym);

  edges->reserve(2*num_asym);
  
  for (int i=0;i<2*num_asym;i+=2) {
    if (open.find(Edge(asym[i],asym[i+1])) == open.end()) {
      edges->push_back(asym[i]);
      edges->push_back(asym[i+1]);
      edges->push_back(asym[i+1]);
      edges->push_back(asym[i]);

      //if ((asym[i] == 7) || (asym[i+1] == 7))
      //  fprintf(stdout,"Edge %d %d\n",asym[i],asym[i+1]);

      open.insert(Edge(asym[i],asym[i+1]));
    }
    else 
      open.erase(Edge(asym[i],asym[i+1]));   
  }
 // fprintf(stderr,"num_asym is %u\n",num_asym);
  delete[] asym;
  
  return;
  }
  //return;

}






std::vector< std::tuple<ngl::IndexType, ngl::IndexType, float>> getSymmetricBetaSpectrum(ANNPointSet<float>* points,
                               int kmax, float param)
{
  IndexType* asym;
  float* asym_beta;
  int num_asym;
  std::set<Edge> open;
  getNeighborGraph_beta(points,kmax,param,&asym,&num_asym, &asym_beta);
  
  result_tuple spectrum; 
  for (int i=0;i<2*num_asym;i+=2) {
      if (open.find(Edge(asym[i],asym[i+1])) == open.end()) {
      //fprintf(stderr,"%u %u %f\n",asym[i],asym[i+1],asym_beta[i-(i/2)]);
      //fprintf(stderr,"%u %u %f\n",asym[i+1],asym[i],asym_beta[i-(i/2)]);
      spectrum.push_back( std::tuple<IndexType, IndexType, float>(asym[i],asym[i+1],asym_beta[i-(i/2)]) );
      spectrum.push_back( std::tuple<IndexType, IndexType, float>(asym[i+1],asym[i],asym_beta[i-(i/2)]) );
      open.insert(Edge(asym[i],asym[i+1]));
                                                            }
   else 
     open.erase(Edge(asym[i],asym[i+1]));   
                                  }
  delete[] asym;
  delete[] asym_beta; 
  return spectrum;
}

