%{
#define SWIG_FILE_WITH_INIT
%}

%include "std_vector.i"
%include "std_string.i"
%include "stdint.i"
%include "numpy.i"

%{
  #include "FunctionHandle.h"
%}


%include "std_vector.i"

%init %{
import_array();
%}


namespace std {
  %template(StringVector) std::vector<std::string>;
  %template(UInt32Vector) std::vector<uint32_t>; // This interferes below
  %template(FHandleVector) std::vector<HDFileFormat::FunctionHandle>;
}

%typemap(in) HDFileFormat::NodeType type
{
  $1 = (HDFileFormat::NodeType)PyLong_AsLong($input);
}

%typemap(in) float
{
  $1 = PyFloat_AsDouble($input);
}



%typemap(in,numinputs=0) std::vector<HDFileFormat::FunctionHandle>* children (std::vector<HDFileFormat::FunctionHandle> tmp)
{
  $1 = &tmp;

  fprintf(stderr,"getFunction-typemap 1\n");
}

%typemap(in,numinputs=0) HDFileFormat::Segmentation& (HDFileFormat::Segmentation tmp)
{
  $1 = &tmp;
}

%typemap(in) const std::vector<std::vector<uint32_t> >* seg (std::vector<std::vector<uint32_t> > tmp)
{
  // Need to add type check
  if (!PyList_Check($input))
    SWIG_exception(SWIG_ValueError,"Expected List object.");


  PyObject* o;

  tmp.resize(PyList_Size($input));

  for (uint32_t i=0;i<tmp.size();i++) {
    o = PyList_GetItem($input,i);

    if (!PyList_Check(o))
      SWIG_exception(SWIG_ValueError,"Expected list of lists object.");

    tmp[i].resize(PyList_Size(o));

    for (uint32_t k=0;k<tmp[i].size();k++)
      tmp[i][k] = PyLong_AsLong(PyList_GetItem(o,k));

  }

  //fprintf(stderr,"Done with segmentation typemap\n");

  $1 = &tmp;
}

%typemap(in) const std::vector<uint32_t>* (std::vector<uint32_t> tmp)
{
  // Need to add type check
  if (!PyList_Check($input))
    SWIG_exception(SWIG_ValueError,"Expected List object.");

  tmp.resize(PyList_Size($input));

  for (uint32_t i=0;i<tmp.size();i++)
    tmp[i] = PyLong_AsLong(PyList_GetItem($input,i));


  //fprintf(stderr,"Done with segmentation typemap\n");

  $1 = &tmp;
}

%typemap(in) uint32_t {
    $1 = PyLong_AsLong($input);
}



/* This isn't working yet because I don't know how to construct a wrapped handle

%typemap(argout) std::vector<HDFileFormat::FunctionHandle>* children
{
  fprintf(stderr,"getFunction-typemap 2\n");
  $result = PyList_New($1->size());

  for (int i=0;i<$1->size();i++) {
    PyList_SET_ITEM($result,i,(*($1))[i]);

  }
}
*/
