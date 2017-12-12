// vim: set syntax=cpp:

/* pypfcommon.i */
%module pypfcommon
%{
  #define SWIG_FILE_WITH_INIT
 /* Includes the header in the wrapper code */
 #include "pypfcommon.h"
 #include "messages.h"

%}

/* Parse the header file to generate wrappers */
%include <numpy.i>

%init %{
  import_array();
%}

%include "messages.h"

// see: http://www.swig.org/Doc1.1/HTML/Python.html and
//  https://gist.github.com/oliora/3607799
%{
  extern PyObject *parser = NULL;
  void SetGridMessageParser(PyObject *cb)
  {
    if (parser != NULL)
      Py_XDECREF(parser);

    assert(cb != Py_None);
    assert(cb != NULL);
    assert(cb != 0);

    parser = cb;

    Py_XINCREF(cb);
  }

  MergeMessageParser(preparser)
  {
    GridMessageMetadata *m = (GridMessageMetadata*) buffer;

    PyObject *arglist = PyTuple_New(2);
    npy_intp dims[3] = {m->nx, m->ny, m->nz};

    PyObject * arr = PyArray_SimpleNewFromData(3, dims, NPY_DOUBLE, (void*)(m+1));
    PyTuple_SET_ITEM(arglist, 0, arr);

    PyObject * it = SWIG_NewPointerObj((void*)m, SWIGTYPE_p_GridDefinition, 0);
    PyTuple_SET_ITEM(arglist, 1, it );
    // REM: typenames cann be looked up in the _wrap.c file

    PyObject *result = PyEval_CallObject(parser, arglist);

    Py_DECREF(arglist);
    Py_XDECREF(result);

    return sizeof(GridMessageMetadata) + sizeof(double) * m->nx * m->ny * m->nz;
  }

  void run()
  {
    // be sure the parser was set before we call run.
    assert(parser);
    _run();
  }

%}

%typemap(in) PyObject * {
    $1 = $input;
}

%typemap(out) PyObject * {
    $result = $1;
}

// exports:
void SetGridMessageParser(PyObject *cb);
void SendSteerMessage(const Action action, const Variable variable,
    int ix, int iy, int iz,
    double *IN_ARRAY3, int DIM1, int DIM2, int DIM3);
void run();


