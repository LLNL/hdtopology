/*
 * Graph.h
 *
 *  Created on: Oct 22, 2018
 *      Author: maljovec
 */

#ifndef NGL_GRAPH_H
#define NGL_GRAPH_H
#include "SearchIndex.h"
#include <cstdlib>

struct Edge {
    int indices[2];
    float distance;
};

class Graph
{
public:

    Graph(SearchIndex *index=NULL,
          int maxNeighbors=-1,
          bool relaxed=false,
          float beta=1.0,
          float p=2.0,
          int discreteSteps=-1,
          int querySize=-1);
    void build(float *X, int N, int D);
    void populate();
    void populate_chunk(int startIndex);
    void populate_whole();
    void push_edges(int *edges, float *distances, int *indices);
    void restart_iteration();
    Edge next();

    virtual ~Graph();
private:
    void advanceIteration();

    float *mData;
    int mCount;
    int mDim;

    int mMaxNeighbors;
    bool mRelaxed;
    float mBeta;
    float mLp;
    float mDiscreteSteps;
    int mQuerySize;

    SearchIndex *mSearchIndex;
    bool mSelfConstructedIndex = false;

    int *mEdges;
    float *mDistances;
    int mRowOffset = 0;
    int mCurrentRow = 0;
    int mCurrentCol = 0;
    bool mReversed = false;
    bool mChunked = false;
    bool mIterationFinished = false;
};

#endif
