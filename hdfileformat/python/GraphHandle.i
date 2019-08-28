%{
	#include "GraphHandle.h"
%}

%include "GraphHandle.h"



%extend HDFileFormat::GraphHandle {
  void setEdgePairs(PyObject* obj) {
    
    PyArrayObject* array;
    if (PyArray_Check(obj)) {
      array = PyArray_GETCONTIGUOUS((PyArrayObject*)(obj));
    }
    else {
      fprintf(stderr,"Does not look like PyArrayObject\n");
      exit(0);
    }
      
    // Set the number of samples
    fprintf(stderr,"GraphHandle:   Found %ld edges\n",PyArray_DIM(array,0));
    $self->sampleCount(PyArray_DIM(array,0));
    
    
    // Pass on the dimensions
    uint32_t dim = PyArray_DIM(array,1);
    fprintf(stderr,"                    of dimension %d\n",dim);
    $self->dimension(dim);
    
     // Set the value size
    PyArray_Descr *descr = PyArray_DTYPE(array);
    fprintf(stderr,"                    with %d bytes each\n",descr->elsize);
 
    $self->setEdgePairs((uint32_t*)PyArray_DATA(array),PyArray_DIM(array,0));
  
   }
 }
