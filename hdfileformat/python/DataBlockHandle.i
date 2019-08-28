%{
	#include "DataBlockHandle.h"
%}

%include "DataBlockHandle.h"



%extend HDFileFormat::DataBlockHandle {

  PyObject* getData() {
    
    // Determine the shape of the 
    npy_intp count = $self->sampleCount();
  
    npy_intp dims[2];
    dims[0] = $self->sampleCount();
    dims[1] = $self->dimension();
    
    int typenum;
    fprintf(stderr,"DataBlockHandle::getData() found type \"%s\"\n",$self->dataType().c_str());
    typenum = numpy_type_num($self->dataType());
      
    
    PyObject* array = PyArray_SimpleNew(2, dims,typenum);    


    // And finally read the data from the file
    $self->readData(PyArray_DATA(array));

    return array;
  }
 }
