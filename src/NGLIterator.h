/*
 * NGLIterator.h
 *
 *  Created on: Oct 24, 2018
 *      Author: maljovec
 */

#ifndef NGLITERATOR_H
#define NGLITERATOR_H

#include "Graph.h"
#include "SearchIndex.h"
#include "EdgeIterator.h"
#include <cstdlib>

//! Implementation of an iterator using a Neighborhood as source
class NGLIterator : public EdgeIterator
{
  public:
    //! Constructor
    NGLIterator(float *data, int N, int D, int maxNeighbors = -1,
                bool relaxed = false, float beta = 1., float p = 2., int discreteSteps=-1,
                int querySize=-1, SearchIndex *knnSearchIndex=NULL)
    {
        graph = new Graph(knnSearchIndex, maxNeighbors, relaxed, beta, p, discreteSteps,
                          querySize);
        graph->build(data, N, D);
        // Load up the first edge
        (*this)++;
    }

    //! Destructor
    ~NGLIterator()
    {
        if (graph != NULL)
        {
            delete graph;
        }
    }

    //! Advance the iterator
    EdgeIterator &operator++(int i)
    {
        Edge e = graph->next();
        if (e.indices[0] != -1 && e.indices[1] != -1)
        {
            currentEdge[0] = e.indices[0];
            currentEdge[1] = e.indices[1];
            currentLength = e.distance;
            finished = false;
        }
        else
        {
            finished = true;
        }
        return *this;
    }

    //! Determine whether we have processed all edges
    bool end()
    {
        return finished;
    }

    //! Return the pointer to an index pair describing the edge
    const uint32_t *operator*() const
    {
        return currentEdge;
    }

    //! Determine whether the edges contain the length
    bool hasLength() const
    {
        return true;
    }

    //! Return the current length or 0 is there is no length
    float length() const
    {
        return currentLength;
    }

    //! Reset the iterator to the beginning
    void reset()
    {
        graph->restart_iteration();
        finished = false;
        (*this)++;
    }

  private:
    Graph *graph;
    uint32_t currentEdge[2];
    float currentLength;
    bool finished;
};

#endif
