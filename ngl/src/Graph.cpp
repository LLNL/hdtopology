/*
 * Graph.h
 *
 *  Created on: Oct 22, 2018
 *      Author: maljovec
 */

#include "Graph.h"
#include "SearchIndex.h"
#include "ngl_cuda.h"
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <unordered_set>
#include <iostream>

Graph::Graph(SearchIndex *index,
             int maxNeighbors,
             bool relaxed,
             float beta,
             float p,
             int discreteSteps,
             int querySize)
    : mMaxNeighbors(maxNeighbors),
      mRelaxed(relaxed),
      mBeta(beta),
      mLp(p),
      mDiscreteSteps(discreteSteps),
      mQuerySize(querySize)
{
    if (index != NULL)
    {
        mSearchIndex = index;
    }
    else
    {
    }
}

void Graph::build(float *X, int N, int D)
{
    mData = X;
    mCount = N;
    mDim = D;

    std::cout << "A" << std::endl;
    mSearchIndex->fit(mData, mCount, mDim);
    std::cout << "B" << std::endl;

    size_t availableGPUMemory = nglcu::get_available_device_memory();
    fprintf(stderr, "availableGPUMemory: %ld\n", availableGPUMemory);
    if (mQuerySize < 0)
    {
        // Because we are using f32 and i32:
        int bytesPerNumber = 4;
        int k = mMaxNeighbors;
        int worstCase;

        // Worst-case upper bound limit of the number of points
        // needed in a single query
        if (mRelaxed)
        {
            // For the relaxed algorithm we need n*D storage for the
            // point locations plus n*k for the representation of the
            // input edges plus another n*k for the output edges
            // We could potentially only need one set of edges for
            // this version
            worstCase = (D + 2) * k;
        }
        else
        {
            // For the strict algorithm, if we are processing n
            // points at a time, we need the point locations of all
            // of their neigbhors, thus in the worst case, we need
            // n + n*k point locations and rows of the edge matrix.
            // the 2*k again represents the need for two versions of
            // the edge matrix. Here, we definitely need an input and
            // an output array
            worstCase = k * (D + 2) * (k + 1);
        }

        // If we are using the discrete algorithm, remember we need
        // to add the template's storage as well to the GPU
        if (mDiscreteSteps > 0)
        {
            availableGPUMemory -= mDiscreteSteps * bytesPerNumber;
        }

        int divisor = bytesPerNumber * worstCase;
        int allocation = (int)( float(availableGPUMemory) / float(divisor));
        mQuerySize = std::min(allocation, mCount);
        //mQuerySize = mCount;
    }

    /*
    if(mQuerySize>100000){
       mQuerySize = 100000;
    }
    */
    fprintf(stderr, "mQuerySize: %ld   mCount: %ld \n", mQuerySize, mCount);
    mChunked = mQuerySize < mCount;

    std::cout << "C" << std::endl;
    populate();
    std::cout << "D" << std::endl;
    // Load up the first edge
    while (mEdges[(mCurrentRow - mRowOffset) * mMaxNeighbors + mCurrentCol] == -1 && !mIterationFinished)
    {
        advanceIteration();
    }
}

void Graph::populate()
{
    if (mChunked)
    {
        mEdges = new int[mQuerySize * mMaxNeighbors];
        mDistances = new float[mQuerySize * mMaxNeighbors];
        populate_chunk(0);
    }
    else
    {
        mEdges = new int[mCount * mMaxNeighbors];
        mDistances = new float[mCount * mMaxNeighbors];
        populate_whole();
    }
}

void Graph::populate_chunk(int startIndex)
{
    mRowOffset = startIndex;
    int count = std::min(mCount - startIndex, mQuerySize);
    int edgeCount = count;
    int endIndex = startIndex + count;

    mSearchIndex->search(mRowOffset, count, mMaxNeighbors, mEdges, mDistances);

    std::unordered_set<int> additionalIndices;
    for (int i = 0; i < count; i++)
    {
        for (int k = 0; k < mMaxNeighbors; k++)
        {
            if (mEdges[i * mMaxNeighbors + k] < startIndex || mEdges[i * mMaxNeighbors + k] >= endIndex)
            {
                additionalIndices.insert(mEdges[i * mMaxNeighbors + k]);
            }
        }
    }

    std::vector<int> indices;
    for (int i = startIndex; i < endIndex; i++)
    {
        indices.push_back(i);
    }

    if (additionalIndices.size() > 0)
    {
        int extraCount = additionalIndices.size();
        for (auto it = additionalIndices.begin(); it != additionalIndices.end(); it++)
        {
            indices.push_back(*it);
        }

        if (!mRelaxed)
        {
            int *extraEdges = new int[extraCount * mMaxNeighbors];
            // This should not be necessary, but otherwise I need to make sure every index
            // properly handles the null pointer.
            float *extraDistances = new float[extraCount * mMaxNeighbors];
            int *extraIndices = indices.data() + count;

            mSearchIndex->search(extraIndices, extraCount, mMaxNeighbors, extraEdges, extraDistances);
            // Again, delete usage of extraDistances if we can make all of the SearchIndices use the following signature:
            // mSearchIndex->search(extraIndices, extraCount, mMaxNeighbors, extraEdges, NULL);
            delete extraDistances;

            edgeCount = count + extraCount;
            int *allEdges = new int[edgeCount * mMaxNeighbors];

            for (int i = 0; i < count; i++)
            {
                for (int k = 0; k < mMaxNeighbors; k++)
                {
                    allEdges[i * mMaxNeighbors + k] = mEdges[i * mMaxNeighbors + k];
                }
            }
            std::unordered_set<int> uniqueIndices(indices.begin(), indices.end());
            additionalIndices.clear();
            for (int i = 0; i < extraCount; i++)
            {
                for (int k = 0; k < mMaxNeighbors; k++)
                {
                    allEdges[(i + count) * mMaxNeighbors + k] = extraEdges[i * mMaxNeighbors + k];
                    additionalIndices.insert(extraEdges[i * mMaxNeighbors + k]);
                }
            }
            delete mEdges;
            delete extraEdges;
            mEdges = allEdges;

            for (auto it = additionalIndices.begin(); it != additionalIndices.end(); it++)
            {
                if(uniqueIndices.find(*it) == uniqueIndices.end()) {
                    indices.push_back(*it);
                }
            }
        }
    }

    float *X = new float[indices.size() * mDim];
    for (int i = 0; i < indices.size(); i++)
    {
        for (int d = 0; d < mDim; d++)
        {
            X[i * mDim + d] = mData[indices[i] * mDim + d];
        }
    }

    if (mDiscreteSteps > 0)
    {
        nglcu::prune_discrete(X, mEdges, indices.data(), indices.size(), mDim, edgeCount,
                              mMaxNeighbors, NULL, mDiscreteSteps, mRelaxed, mBeta, mLp, count);
    }
    else
    {
        nglcu::prune(X, mEdges, indices.data(), indices.size(), mDim, edgeCount,
                     mMaxNeighbors, mRelaxed, mBeta, mLp, count);
    }

    delete X;
}

void Graph::populate_whole()
{
    mSearchIndex->search(0, mCount, mMaxNeighbors, mEdges, mDistances);

    std::cerr << "Relaxed: " << mRelaxed << std::endl;
    if (mDiscreteSteps > 0)
    {
        nglcu::prune_discrete(mData, mEdges, NULL, mCount, mDim, mCount,
                              mMaxNeighbors, NULL, mDiscreteSteps, mRelaxed, mBeta, mLp);
    }
    else
    {
        nglcu::prune(mData, mEdges, NULL, mCount, mDim, mCount,
                     mMaxNeighbors, mRelaxed, mBeta, mLp);
    }
}

void Graph::restart_iteration()
{
    mIterationFinished = false;
    mCurrentCol = 0;
    mCurrentRow = 0;
    // Load up the first edge
    while (mEdges[(mCurrentRow - mRowOffset) * mMaxNeighbors + mCurrentCol] == -1 && !mIterationFinished)
    {
        advanceIteration();
    }
}

Edge Graph::next()
{
    Edge e;
    if (mIterationFinished)
    {
        // Set up the next round of iteration
        mIterationFinished = false;
        e.indices[0] = -1;
        e.indices[1] = -1;
        e.distance = 0;
        return e;
    }
    int currentIndex = (mCurrentRow - mRowOffset) * mMaxNeighbors + mCurrentCol;
    e.distance = mDistances[currentIndex];
    e.indices[0] = mEdges[currentIndex];
    e.indices[1] = mCurrentRow;
    advanceIteration();
    while (mEdges[(mCurrentRow - mRowOffset) * mMaxNeighbors + mCurrentCol] == -1)
    {
        advanceIteration();
    }

    return e;
}

void Graph::advanceIteration()
{
    mCurrentCol++;
    if (mCurrentCol >= mMaxNeighbors)
    {
        mCurrentCol = 0;
        mCurrentRow++;
        if (mCurrentRow >= mCount)
        {
            mCurrentRow = 0;
            mIterationFinished = true;
        }
        // If we have changed rows, let's ensure we don't need to run
        // another query
        if (mChunked)
        {
            if (mCurrentRow - mRowOffset >= mQuerySize || mCurrentRow == 0)
            {
                populate_chunk(mCurrentRow);
            }
        }
    }
}

Graph::~Graph()
{
    if (mSelfConstructedIndex)
    {
        delete mSearchIndex;
    }
    delete mEdges;
    delete mDistances;
}
