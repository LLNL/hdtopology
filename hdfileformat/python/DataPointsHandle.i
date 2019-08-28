%{
	#include "DataPointsHandle.h"
%}

%include "DataPointsHandle.h"

%extend HDFileFormat::DataPointsHandle {
  PyObject* getData() {
    
    // Determine the shape of the 
    npy_intp count = $self->sampleCount();
    
     // Determine the shape of the 
    // Now build the descriptor which is a list of tuples
    PyObject* dtype = PyList_New($self->dimension());

    // Determine the shape of the 
    for (uint32_t i=0;i<$self->dimension();i++) {
      PyObject* tuple = PyTuple_New(2);
      
      // Each tuple contains the attribute name
      PyTuple_SetItem(tuple,0,PyString_FromString($self->attributeName(i).c_str()));
      // And the datatype and here we assume HDFF uses the numpy naming convention
      PyTuple_SetItem(tuple,1,PyString_FromString(numpy_type_name($self->dataType())));
      
      PyList_SetItem(dtype,i,tuple);
    }

    // Determine the shape of the 
    // Now convert that dtype to a descriptor
    PyArray_Descr* descr;

    PyArray_DescrConverter(dtype, &descr);

    // Make sure to clean up the list
    Py_DECREF(dtype);
    
    // Now we can make the basic array 
    PyObject* array = PyArray_SimpleNewFromDescr(1,&count,descr);

    // And finally read the data from the file
    $self->readData(PyArray_DATA(array));

     return array;
  }
 }

 
 %extend HDFileFormat::DataPointsHandle {
  void setData(PyObject* obj) {
    
    PyArrayObject* array;
    if (PyArray_Check(obj)) {
      array = PyArray_GETCONTIGUOUS((PyArrayObject*)(obj));
    }
    else {
      fprintf(stderr,"Does not look like PyArrayObject\n");
      exit(0);
    }
      
    // Set the number of samples
    fprintf(stderr,"DataPointsHandle:   Found %ld points\n",PyArray_DIM(array,0));
    $self->sampleCount(PyArray_DIM(array,0));
    
    
    // Pass on the dimensions
    uint32_t dim = PyArray_DIM(array,1);
    fprintf(stderr,"                    of dimension %d\n",dim);
    $self->dimension(dim);
    
     // Set the value size
    PyArray_Descr *descr = PyArray_DTYPE(array);
    fprintf(stderr,"                    with %d bytes each\n",descr->elsize);
    $self->valueSize(descr->elsize);
    
    // Resize attribute names
    std::vector<std::string> names(dim);
    char tmp[20];
    for (uint32_t i=0;i<dim;i++) {
      sprintf(tmp,"Attribute %d",i);
      names[i] = tmp;
    }
    $self->attributeNames(names);
     
    
    
    /*
    PyObject *list = PyDict_Keys(descr->fields);  
    
    for (uint32_t i=0;i<dim;i++) {
      fprintf(stderr,"names %p ... %p\n",list,PyList_GetItem(descr->fields,i));
      PyList_GetItem(descr->fields,i);
      
      //names[i] = std::string(PyString_AsString(PySequence_GetItem(descr->names,i)));
    } 
    $self->attributeNames(names);
    */
    
    // Set the datatype using numpy strings  (HACK)
    $self->dataType("float32");
    
    // Pass on the pointer (this will *not* copy the data
    $self->setData(PyArray_DATA(array));
    
    
   
   }
 }
 