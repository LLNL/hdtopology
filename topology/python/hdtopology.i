%module(directors="1") hdtopology

%include "typemaps.i"
%include "ExtremumGraph.i"
%include "Flags.i"
#ifdef NGLCU_AVAILABLE
    %include "NGLIterator.i"
#endif
%include "Histogram.i"
%include "JointDistributions.i"
%include "Selectivity.i"
%include "stdint.i"
