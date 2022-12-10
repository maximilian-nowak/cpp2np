#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
#include <string>
#include <array>
#include <iostream>

static PyObject* hello(PyObject* self, PyObject* args){
    std::string s = "Hello Python, this is C++!";
    return Py_BuildValue("s", s.c_str());
}

// Define a PyCapsule_Destructor
void free_wrap(PyObject *capsule){
    void * memory = PyCapsule_GetPointer(capsule, PyCapsule_GetName(capsule));
    std::cout << "destruction" << std::endl;
    free(memory);
};

static PyObject* cpp2np_wrap(PyObject* self, PyObject* args, PyObject* kwargs){
    PyObject* arr = NULL;
    PyObject* input = NULL;
    npy_int owndata=1;
    PyArray_Descr* dtype = NULL;

    std::string keys[] = {"input", "owndata", "dtype"};
    static char* kwlist[] = {keys[0].data(), keys[1].data(), keys[2].data(), NULL};
    if(!PyArg_ParseTupleAndKeywords(args, kwargs, "O|iO", kwlist, &input, &owndata, &dtype)){
        return NULL;
    }
    
    if(dtype && !PyArray_DescrCheck(dtype)) {
        return NULL;
    }
    
    PyObject* memview = PyMemoryView_GetContiguous(input, PyBUF_READ, 'C');
    if(!memview) return NULL;
    PyObject* err = PyErr_Occurred();
	if (err) {
        PyErr_Clear();
	    return NULL;
	}

    Py_buffer* buffer = NULL;
    if(PyMemoryView_Check(memview)) {
        buffer = PyMemoryView_GET_BUFFER(memview);
    }
    if(!buffer) {
        Py_XDECREF(buffer);
        Py_XDECREF(memview);
        return NULL;
    }
    arr = PyArray_SimpleNewFromData(buffer->ndim, buffer->shape, dtype ? dtype->type_num : NPY_INT, buffer->buf);

    // array holds ref to buffer through this wrapper
    if(arr && buffer && owndata) {
        Py_INCREF(input);
        // set array as owner of data buffer
        PyArray_ENABLEFLAGS((PyArrayObject*) arr, NPY_ARRAY_OWNDATA | NPY_ARRAY_WRITEABLE );
        if(PyArray_SetBaseObject((PyArrayObject*) arr, input) < 0) {
            Py_DECREF(arr);
            Py_XDECREF(buffer);
            Py_XDECREF(memview);
            Py_XDECREF(input);
            return NULL;
        }
        //alternative way: assigning a destructor as base object
        // PyObject* capsule = PyCapsule_New(input, "cpp_buffer", (PyCapsule_Destructor)&free_wrap);
        // if (PyArray_SetBaseObject((PyArrayObject*)arr, capsule) == -1) {
        //     Py_DECREF(arr);
        //     Py_XDECREF(buffer);
        //     Py_XDECREF(memview);
        //     Py_XDECREF(input);
        //     return NULL;
        // } 
    }
    return arr ? arr : NULL;
}

static char cpp2np_docs[] = {
    "C Numpy extension.\n"
};

static PyMethodDef cpp2np_funcs[] = {
    {"hello", (PyCFunction)hello, METH_VARARGS, "Hello World"},
    {"wrap", (PyCFunction)cpp2np_wrap, METH_VARARGS | METH_KEYWORDS, "create numpy array around external data pointer"},
    {nullptr, nullptr, 0, nullptr}
};

static struct PyModuleDef cpp2npmodule = {
    PyModuleDef_HEAD_INIT,
    "cpp2np",
    cpp2np_docs,
    -1,
    cpp2np_funcs
};

PyMODINIT_FUNC PyInit_cpp2np(void){
    import_array();
    return PyModule_Create(&cpp2npmodule);
}
