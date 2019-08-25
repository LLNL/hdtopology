
%{
#define SWIG_FILE_WITH_INIT
%}


%include "numpy.i"
%include "exception.i"

%init %{
import_array();
%}


%typemap(in) (ngl::ANNPointSet<float>* ) (ngl::ANNPointSet<float> tmp){

  PyArrayObject* data_ptr = PyArray_GETCONTIGUOUS((PyArrayObject*)$input);
  
  if (PyArray_TYPE(data_ptr) != NPY_FLOAT32)
    SWIG_exception(SWIG_ValueError,"Expected array of float32.");

  // Set the current dimension 
  ngl::Geometry<float>::init(PyArray_DIM(data_ptr,1));

  // Read in the number of points
  tmp.loadData((float*)PyArray_DATA(data_ptr),PyArray_DIM(data_ptr,0));
  
  $1 = &tmp;
}


%typemap(in,numinputs=0) (ngl::IndexType** edges, int* numEdges) (ngl::IndexType* tmp_edges,int count=0)
{
  $1 = &tmp_edges;
  $2 = &count;
}

%typemap(in,numinputs=0) (std::vector<ngl::IndexType>* edges) (std::vector<ngl::IndexType> tmp_edges)
{
  $1 = &tmp_edges;
}

%typemap(argout)  (ngl::IndexType** edges, int* numEdges) {
  
  if (*$1 == NULL)
    SWIG_exception(SWIG_ValueError,"No output values to convert");
    
  npy_intp dim[2];
  dim[0] = *$2;
  dim[1] = 2;
  $result = PyArray_SimpleNew(2, dim, PyArray_UINT);
  memcpy(PyArray_DATA($result),*$1,sizeof(unsigned int)*dim[0]*dim[1]);
}

%typemap(argout)  (std::vector<ngl::IndexType>* edges) {
  
  if ($1 == NULL)
    SWIG_exception(SWIG_ValueError,"No output values to convert");
    
  npy_intp dim[2];
  dim[0] = $1->size()/2;
  dim[1] = 2;
  $result = PyArray_SimpleNew(2, dim, PyArray_UINT);
  memcpy(PyArray_DATA($result),&(*$1)[0],sizeof(unsigned int)*dim[0]*dim[1]);
}

