#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
#include <string>
#include <array>
#include <iostream>
#include <memory>

static PyObject* hello(PyObject* self, PyObject* args){
    std::string s = "Hello Python, this is C++!";
    return Py_BuildValue("s", s.c_str());
};

/// Define a PyCapsule_Destructor for wrapper. merely serves as debug information.
void decref_capsule(PyObject *capsule){
    // void* memory = (void*) PyCapsule_GetPointer(capsule, PyCapsule_GetName(capsule));
    std::cout << "delete pyobject reference" << std::endl;
};

/// Define a PyCapsule_Destructor for capsule
void free_capsule(PyObject *capsule){
    void * memory = (void*) PyCapsule_GetPointer(capsule, PyCapsule_GetName(capsule));
    std::cout << "free memory buffer" << std::endl;
    free(memory);
};

static PyObject* cpp2np_wrap(PyObject* self, PyObject* args, PyObject* kwargs){
    npy_intp ptr;  // npy_intp => intptr_t => unsigned long it
    npy_int free_mem_on_del=0;  // if true frees buffer after deletion of numpy array

    PyObject* arr = NULL;
    PyObject* in_shape = NULL;
    PyArray_Descr* dtype = NULL;
    
    std::string keys[] = {"input", "shape", "dtype", "free_mem_on_del"};
    static char* kwlist[] = {keys[0].data(), keys[1].data(), keys[2].data(), keys[3].data(), NULL};
    if(!PyArg_ParseTupleAndKeywords(args, kwargs, "lO|Oi", kwlist, &ptr, &in_shape, &dtype, &free_mem_on_del)){
        return NULL;
    }
    if(dtype && !PyArray_DescrCheck(dtype)) {
        return NULL;
    }
    if(!PyTuple_Check(in_shape)) {
        return NULL;
    }
    
    // convert python tuple to int[]
    Py_ssize_t ndim = PyTuple_GET_SIZE(in_shape);
    Py_ssize_t shape[ndim];
    for(int i=0; i<ndim; ++i){
        PyObject* pval = PyTuple_GET_ITEM(in_shape, i);
        shape[i] = PyLong_AsLong(pval);
    }
    
    void* buf = (void*) ptr;
    arr = PyArray_SimpleNewFromData(ndim, shape, dtype ? dtype->type_num : NPY_INT, buf);

    // free memory by binding destructor capsule to numpy array
    // warning: this won't take into account the reference count on the buffer
    if(free_mem_on_del) {
        PyObject* capsule = PyCapsule_New(buf, "buffer_capsule", (PyCapsule_Destructor) &free_capsule);
        if (PyArray_SetBaseObject((PyArrayObject*)arr, capsule) == -1) {
            Py_DECREF(arr);
            return NULL;
        }
    }
    return arr;
};

static PyObject* cpp2np_array_2x2(PyObject* self, PyObject* args){
    auto* data = new std::array<std::array<npy_int, 2>, 2>({{{1,2},{3,4}}});

    npy_intp ptr = (npy_intp) data;
    std::cout << ptr << std::endl;

    // PyObject* ret = PyTuple_New(1);
    // PyTuple_SET_ITEM(ret, 0, PyLong_FromLong(ptr));

    return PyLong_FromLong(ptr);
};

static PyObject* cpp2np_descr(PyObject* self, PyObject* args, PyObject* kwargs){
    PyObject* input = NULL;
    PyArrayObject* arr = NULL;
    PyObject* ret = NULL;
    
    std::string keys[] = {"input"};
    static char* kwlist[] = {keys[0].data(), NULL};
    if(!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &input)){
        return NULL;
    }

    if(PyObject_HasAttr(input, PyUnicode_FromString("__array_interface__"))) {
        PyObject* interf = PyObject_GetAttr(input, PyUnicode_FromString("__array_interface__"));
        return interf;
    }

    arr = (PyArrayObject*) input;

    ret = PyDict_New();
    npy_intp ptr = (npy_intp) PyArray_DATA(arr);
    Py_ssize_t ndim = PyArray_NDIM(arr);
    npy_intp* shape = PyArray_SHAPE(arr);
    PyArray_Descr* dtype = PyArray_DTYPE(arr);

    PyDict_SetItem(ret, PyUnicode_FromString("ptr"), PyLong_FromLong(ptr));
    PyDict_SetItem(ret, PyUnicode_FromString("ndim"), PyLong_FromLong(ndim));

    if(shape) {
        PyObject* shape_tuple = PyTuple_New(ndim);
        for(int i=0; i<ndim; ++i) {
            PyTuple_SET_ITEM(shape_tuple, i, PyLong_FromLong(shape[i]));
        }
        PyDict_SetItem(ret, PyUnicode_FromString("shape"), shape_tuple);
    }

    if(PyArray_DescrCheck(dtype)) {
        PyDict_SetItem(ret, PyUnicode_FromString("typenum"), PyLong_FromLong(dtype->type_num));
    }

    return ret;
};

static PyObject* cpp2np_freemem(PyObject* self, PyObject* args, PyObject* kwargs){
    npy_intp ptr;
    std::string keys[] = {"pointer"};
    static char* kwlist[] = {keys[0].data(), NULL};
    if(!PyArg_ParseTupleAndKeywords(args, kwargs, "l", kwlist, &ptr)){
        return Py_BuildValue("i", 0);
    }

    std::cout << "destruction" << std::endl;

    void* buf = (void*) ptr;
    free(buf);
    
    return Py_BuildValue("i", 1);
};

static char cpp2np_docs[] = {
    "C Numpy extension.\n"
};

static PyMethodDef cpp2np_funcs[] = {
    {"hello", (PyCFunction)hello, METH_VARARGS, "Hello World"},
    {"array_2x2", (PyCFunction)cpp2np_array_2x2, METH_VARARGS | METH_KEYWORDS, "create test array from c++ memory"},
    {"wrap", (PyCFunction)cpp2np_wrap, METH_VARARGS | METH_KEYWORDS, "create numpy array from pointer"},
    {"descr", (PyCFunction)cpp2np_descr, METH_VARARGS | METH_KEYWORDS, "return array interface as dict"},
    {"freemem", (PyCFunction)cpp2np_freemem, METH_VARARGS | METH_KEYWORDS, "free the memory the pointer is referencing"},
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
};
