// vim: set syntax=cpp:

/* pypfAnalyzer.i */
%module pypfAnalyzer
%{
  #define SWIG_FILE_WITH_INIT
 /* Includes the header in the wrapper code */
 #include "pypfAnalyzer.h"
 #include "messages.h"
 #include <signal.h>

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

    // low: assert cb is an function object!

    parser = cb;

    Py_XINCREF(cb);
  }

  MergeMessageParser(preparser)
  {
    GridMessageMetadata *m = (GridMessageMetadata*) buffer;

    PyObject *arglist = PyTuple_New(2);
    npy_intp dims[3] = {m->nz, m->ny, m->nx};

    PyObject * arr = PyArray_SimpleNewFromData(3, dims, NPY_DOUBLE, (void*)(m+1));
    PyTuple_SET_ITEM(arglist, 0, arr);

    PyObject * it = SWIG_NewPointerObj((void*)m, SWIGTYPE_p_GridMessageMetadata, 0);
    PyTuple_SET_ITEM(arglist, 1, it );
    // REM: typenames can be looked up in the _wrap.c file

    PyObject *result = PyEval_CallObject(parser, arglist);
    PyObject *error =  PyErr_Occurred();

    if (error != NULL)
    {
      PyErr_Print();

      raise(SIGABRT);
    }

    Py_DECREF(arglist);
    Py_XDECREF(result);

    return sizeof(GridMessageMetadata) + sizeof(double) * m->nx * m->ny * m->nz;
  }

  extern PyObject *onInit = NULL;
  void SetOnInit(PyObject *cb)
  {
    if (onInit != NULL)
      Py_XDECREF(onInit);

    assert(cb != Py_None);
    assert(cb != NULL);
    assert(cb != 0);

    // low: assert cb is an function object!

    onInit = cb;

    Py_XINCREF(cb);
  }

  void callOnInit()
  {
    if (onInit == NULL)
      return;

    // call onInit!
    PyObject *result = PyEval_CallObject(onInit, NULL);
    PyObject *error =  PyErr_Occurred();

    if (error != NULL)
    {
      PyErr_Print();

      raise(SIGABRT);
    }

    Py_XDECREF(result);
  }

  void run(char *logstamps[], size_t logstampsc)
  {
    // be sure the parser was set before we call run.
    assert(parser);
    _run(logstamps, logstampsc);
  }

%}

%typemap(in) PyObject * {
    $1 = $input;
}

%typemap(out) PyObject * {
    $result = $1;
}

%typemap(in) (char *logstamps[], size_t logstampsc) {
  assert($input != Py_None);
  int i;
  if (!PyList_Check($input)) {
    PyErr_SetString(PyExc_ValueError, "Expecting a list");
    return NULL;
  }
  $2 = PyList_Size($input);
  $1 = (char **) malloc($2 * sizeof(char *));
  for (i = 0; i < $2; i++) {
    PyObject *s = PyList_GetItem($input,i);
    if (!PyString_Check(s)) {
        free($1);
        PyErr_SetString(PyExc_ValueError, "List items must be strings");
        return NULL;
    }
    $1[i] = PyString_AsString(s);
  }
}

%typemap(freearg) (char *logstamps[], size_t logstampsc) {
   if ($1) free($1);  // REM: if some error in this line occurs, see if everything in the python was correct!
}

%typemap(in) (StampLog slog[], size_t n) {
  printf("Mapping SendLog\n");
  assert($input != Py_None);
  int i;
  if (!PyList_Check($input)) {
    PyErr_SetString(PyExc_ValueError, "Expecting a list");
    return NULL;
  }
  $2 = PyList_Size($input);
  $1 = (StampLog *) malloc(($2)*sizeof(StampLog));
  for (i = 0; i < $2; i++) {
    PyObject *s = PyList_GetItem($input,i);
    StampLog * tmp;
    if (SWIG_ConvertPtr(s, (void **) &tmp, SWIGTYPE_p_StampLog, SWIG_POINTER_EXCEPTION) == -1) {
      free($1);
      PyErr_SetString(PyExc_ValueError, "List items must be StampLogs");
      return NULL;
    }
    memcpy(&($1[i]), tmp, sizeof(StampLog));
  }
}

%typemap(freearg) (StampLog slog[], size_t n) {
   if ($1) free($1);
}

// exports:
void SetGridMessageParser(PyObject *cb);
void SetOnInit(PyObject *cb);
void SendSteerMessage(const Action action, const Variable variable,
    int ix, int iy, int iz,
    double *IN_ARRAY3, int DIM1, int DIM2, int DIM3);
void run(char *logstamps[], size_t logstampsc);
void SendLog(StampLog slog[], size_t n);
