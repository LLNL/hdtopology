%module typemaps
%{
#define SWIG_FILE_WITH_INIT
%}

%include "std_vector.i"
%include "stdint.i"
%include "numpy.i"
%include "std_string.i"
%include "std_pair.i"
%include "stl.i"
%include "std_set.i"

%init %{
import_array();
%}


// on the python side the Neighborhood should be a numpy array of uint32_t
%typemap(in) (const Neighborhood* edges) (Neighborhood neigh){

  /*
  PyArrayObject* edges = PyArray_GETCONTIGUOUS((PyArrayObject*)PyObject_GetAttrString($input,"edges"));

  if (PyArray_TYPE(edges) != NPY_UINT32)
    SWIG_exception(SWIG_ValueError,"Expected list of uint32_t edges.");
  */

  neigh.data((uint32_t*)PyArray_DATA($input));
  neigh.size(PyArray_DIM($input,0));


  if (PyObject_HasAttrString($input,"length") != 0) {
    PyArrayObject* length = PyArray_GETCONTIGUOUS((PyArrayObject*)PyObject_GetAttrString($input,"length"));

    if (PyArray_TYPE(length) != NPY_FLOAT32)
      SWIG_exception(SWIG_ValueError,"Expected list of float values.");

    if (neigh.size() !=  PyArray_DIM(length,0))
      SWIG_exception(SWIG_ValueError,"Number of edges and number of length values don't match.");

    neigh.setLength((float*)PyArray_DATA(length));
  }

  $1 = &neigh;
}

// on the python side, the HDData should be recarray
%typemap(in) (const HDData*) (HDData tmp){

  //if (PyObject_HasAttrString($input,"attributes") == 0)
  //  SWIG_exception(SWIG_ValueError,"Expected HDData object.");

  PyArrayObject* data_ptr = (PyArrayObject*)$input;


  tmp.data((float*)PyArray_DATA(data_ptr));
  tmp.size(PyArray_DIM(data_ptr,0));
  tmp.dim(PyArray_STRIDE(data_ptr,0) / 4);

  //assume the last column is the function and there is only one function
  tmp.func(tmp.dim()-1);
  tmp.attr(tmp.dim());

  PyArray_Descr *dtype;
  PyObject *names, *name;

  std::vector<std::string> attributes(tmp.dim());
  dtype = PyArray_DTYPE(data_ptr);
  names = dtype->names;

  if (names != NULL) {
        names = PySequence_Fast(names, NULL);
        for (uint32_t i=0;i<tmp.dim();i++) {
          name = PySequence_Fast_GET_ITEM(names, i);
          PyObject* repr = PyObject_Repr(name);
          PyObject* str = PyUnicode_AsEncodedString(repr, "utf-8", "E");
          const char *bytes = PyBytes_AS_STRING(str);
          printf("REPR: %s\n", bytes);
          attributes[i] = std::string(bytes);
        }
      }

  tmp.attributes(attributes);
  $1 = &tmp;

  //if (PyArray_TYPE(data_ptr) != NPY_FLOAT32)
  //  SWIG_exception(SWIG_ValueError,"Expected array of float32.");

}

%typemap(in) (const Flags*) (Flags tmp){

  PyArrayObject* data_ptr = (PyArrayObject*)$input;
  if (data_ptr->dimensions[0] <= 1)
    tmp = Flags(NULL);
  else {

    PyArrayObject* data_ptr = (PyArrayObject*)$input;
    tmp = Flags((uint8_t*)PyArray_DATA(data_ptr));
  }

  $1 = &tmp;
 }


// typecheck helps decode this as a parameter correctly when used in overloaded
// function signatures
%typemap(typecheck) const HDData* {
    $1 = PyArray_Check($input) ? 1 : 0;
}

%typemap(typecheck) const Neighborhood* {
    $1 = PyArray_Check($input) ? 1 : 0;
}

%typemap(typecheck) const Flags* {
    $1 = PyArray_Check($input) ? 1 : 0;

    if ($1) {
      PyArrayObject* data_ptr = (PyArrayObject*)$input;

      $1 = $1 && (PyArray_TYPE(data_ptr) == NPY_UINT8 );
    }
}

%apply const HDData* {const HDFunction*};

%typemap(out) std::vector<uint32_t> {

  npy_intp dim = $1.size();
  $result = PyArray_SimpleNew(1, &dim, PyArray_UINT32);
  memcpy(PyArray_DATA($result),&((*(&$1))[0]),sizeof(uint32_t)*dim);
}


%typemap(out) std::vector<float> {

  npy_intp dim = $1.size();
  $result = PyArray_SimpleNew(1, &dim, PyArray_FLOAT32);
  memcpy(PyArray_DATA($result),&((*(&$1))[0]),sizeof(float)*dim);
}


%typemap(out) std::vector<std::vector<uint32_t> > {

  PyObject *seg;
  npy_intp dim = $1.size();
  npy_intp count;

  $result = PyList_New(dim);

  //fprintf(stderr,"In typemap vector vector\n");

  for (uint32_t i=0;i<dim;i++) {
    count = $1.at(i).size();
    seg = PyArray_SimpleNew(1,&count , PyArray_UINT32);
    memcpy(PyArray_DATA(seg),&($1.at(i).at(0)),sizeof(uint32_t)*count);

    PyList_SetItem($result,i,seg);
  }
}

%typemap(out) std::vector<std::vector<float> > {

  PyObject *seg;
  npy_intp dim = $1.size();
  npy_intp count;

  $result = PyList_New(dim);

  //fprintf(stderr,"In typemap vector vector\n");

  for (uint32_t i=0;i<dim;i++) {
    count = $1.at(i).size();
    seg = PyArray_SimpleNew(1,&count , PyArray_FLOAT32);
    memcpy(PyArray_DATA(seg),&($1.at(i).at(0)),sizeof(float)*count);

    PyList_SetItem($result,i,seg);
  }
}


%typemap(in) uint32_t {
    $1 = (uint32_t) PyLong_AsLong($input);
}

/*
%typemap(out) uint32_t {
    $result = PyLong_FromLong((long) $1);
}

%typemap(in) int32_t {
	$1 = (int) PyLong_AsLong($input);
}

%typemap(in) int {
	$1 = (int) PyLong_AsLong($input);
}
*/

/*
%typemap(out) Histogram {

   std::vector<npy_intp> dims(($1).dimension(),($1).resolution());

   $result = PyArray_SimpleNewFromData(($1).dimension(),&dims[0],
                                       PyArray_UINT32,($1).data());
 }
*/



//I'm using a double vector here only because using a float
// messes up the above declaration, I really don't need more
// precision, but it was the quickest way to get this working
// Admittedly, this is an ugly hack
namespace std {
  %template(DoubleVector) vector<double>;
  %template(StringVector) vector<string>;
  %template(FloatPair) pair<float,float>;
  %template(FloatPairVector) vector<pair<float, float> >;
  %template(FloatVectorVector) vector<vector<float> >;
}

%apply (float* IN_ARRAY1, int DIM1) {(const float *x_data, int x_dim),
    (const float *y_data, int y_dim)};
%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(const float *data, int y_dim, int x_dim),
    (float* data, int N, int D)};
%apply (unsigned char* IN_ARRAY2, int DIM1, int DIM2) {(const unsigned char*colors, int c_dim, int channels)};
