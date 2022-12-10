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
}

/// Define a PyCapsule_Destructor for wrapper. merely serves as debug information.
void decref_capsule(PyObject *capsule){
    // void* memory = (void*) PyCapsule_GetPointer(capsule, PyCapsule_GetName(capsule));
    std::cout << "decremented reference count" << std::endl;
};
/// Define a PyCapsule_Destructor for capsule
void free_capsule(PyObject *capsule){
    void * memory = (void*) PyCapsule_GetPointer(capsule, PyCapsule_GetName(capsule));
    std::cout << "destruction" << std::endl;
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

    PyCapsule_Destructor* destr = (PyCapsule_Destructor*) ((free_mem_on_del) ? &free_capsule : &decref_capsule);
    PyObject* capsule = PyCapsule_New(buf, "wrapper_capsule", (PyCapsule_Destructor) destr);
    if (PyArray_SetBaseObject((PyArrayObject*)arr, capsule) == -1) {
        Py_DECREF(arr);
        return NULL;
    }
    return arr;
}

static PyObject* cpp2np_wrapFromCapsule(PyObject* self, PyObject* args, PyObject* kwargs){
    PyObject* arr = NULL;
    PyObject* input;
    PyObject* in_shape = NULL;
    npy_int free_mem_on_del=0;
    PyArray_Descr* dtype = NULL;

    std::cout << "test" << std::endl;
    
    std::string keys[] = {"input", "shape", "dtype", "free_mem_on_del"};
    static char* kwlist[] = {keys[0].data(), keys[1].data(), keys[2].data(), keys[3].data(), NULL};
    if(!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|Oi", kwlist, &input, &in_shape, &dtype, &free_mem_on_del)){
        return NULL;
    }
    if(dtype && !PyArray_DescrCheck(dtype)) {
        return NULL;
    }
    if(!PyTuple_Check(in_shape)) {
        return NULL;
    }
    
    Py_ssize_t ndim = PyTuple_GET_SIZE(in_shape);
    Py_ssize_t shape[ndim];
    
    std::cout << ndim << std::endl;
    
    for(int i=0; i<ndim; ++i){
        PyObject* pval = PyTuple_GET_ITEM(in_shape, i);
        shape[i] = PyLong_AsLong(pval);
        std::cout << shape[i] << std::endl;
    }
    
    if(PyCapsule_IsValid(input, "cpointer")){
        void* buf = (void*) PyCapsule_GetPointer(input, "cpointer");
        arr = PyArray_SimpleNewFromData(ndim, shape, dtype ? dtype->type_num : NPY_INT, buf);

        PyCapsule_Destructor* destr = (PyCapsule_Destructor*) ((free_mem_on_del) ? &free_capsule : &decref_capsule);
        PyObject* capsule = PyCapsule_New(buf, "wrapper_capsule", (PyCapsule_Destructor) destr);
        if (PyArray_SetBaseObject((PyArrayObject*)arr, capsule) == -1) {
            Py_DECREF(arr);
            return NULL;
        }
        return arr;
    }
    return NULL;
}

static PyObject* cpp2np_array(PyObject* self, PyObject* args){
    // PyObject *arr;

    // int shape = 1;
    // int dims[] = {3};
    auto* data = new std::array<npy_int, 3>({1, 2, 3});
    
    // arr = PyArray_SimpleNewFromData(shape, dims, NPY_DOUBLE, (void *)data);

    // PyObject *capsule = PyCapsule_New(data, "cpointer", (PyCapsule_Destructor)&free_wrap);
    // PyArray_SetBaseObject((PyArrayObject *) arr, capsule);

    // PyObject* ret = PyDict_New();
    // PyDict_SetItem(ret, PyUnicode_FromString("ptr"), PyLong_FromLong(ptr));

    npy_intp ptr = (npy_intp) data;
    std::cout << ptr << std::endl;

    PyObject* ret = PyTuple_New(1);
    PyTuple_SET_ITEM(ret, 0, PyLong_FromLong(ptr));
    
    return ret;
}

static PyObject* cpp2np_arrayCapsule(PyObject* self, PyObject* args){
    // PyObject *arr;

    // int shape = 1;
    // int dims[] = {3};
    auto* data = new std::array<npy_int, 3>({1, 2, 3});
    
    // arr = PyArray_SimpleNewFromData(shape, dims, NPY_DOUBLE, (void *)data);

    PyObject *capsule = PyCapsule_New(data, "cpointer", (PyCapsule_Destructor)&free_capsule);
    // PyArray_SetBaseObject((PyArrayObject *) arr, capsule);
    
    return capsule;
}

static PyObject* cpp2np_getDescr(PyObject* self, PyObject* args, PyObject* kwargs){
    PyObject *arr = NULL;
    PyObject* ret = NULL;
    
    std::string keys[] = {"input"};
    static char* kwlist[] = {keys[0].data(), NULL};
    if(!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &arr)){
        return NULL;
    }
    
    ret = PyDict_New();

    npy_intp ptr = (npy_intp) PyArray_DATA((PyArrayObject*)arr);
    
    PyDict_SetItem(ret, PyUnicode_FromString("ptr"), PyLong_FromLong(ptr));
    PyDict_SetItem(ret, PyUnicode_FromString("ndim"), PyLong_FromLong(PyArray_NDIM((PyArrayObject*)arr)));

    // TODO: add shape and type to dict
    // npy_intp *PyArray_SHAPE(PyArrayObject *arr)
    // PyArray_Descr *PyArray_DTYPE(PyArrayObject* arr)
    
    return ret;
}

static PyObject* cpp2np_free(PyObject* self, PyObject* args, PyObject* kwargs){
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
}


static PyObject* cpp2np_wrap2(PyObject* self, PyObject* args, PyObject* kwargs){
    PyObject* arr = NULL;
    PyObject* input = NULL;
    npy_int owndata=1;
    PyArray_Descr* dtype = NULL;

    std::cout << "test" << std::endl;
    std::string keys[] = {"input", "owndata", "dtype"};
    static char* kwlist[] = {keys[0].data(), keys[1].data(), keys[2].data(), NULL};
    if(!PyArg_ParseTupleAndKeywords(args, kwargs, "O|iO", kwlist, &input, &owndata, &dtype)){
        return NULL;
    }
    
    std::cout << "test" << std::endl;
    if(dtype && !PyArray_DescrCheck(dtype)) {
        return NULL;
    }
    std::cout << "test" << std::endl;

    PyObject* memview = NULL;
    if(PyCapsule_IsValid(input, "cpointer")){
        std::cout << "valid" << std::endl;
        void* memory = (void*) PyCapsule_GetPointer(input, "cpointer");
        const npy_intp shape[] = {3};
        arr = PyArray_SimpleNewFromData(1, shape, NPY_INT, memory);

        PyObject* capsule = PyCapsule_New(input, "cpp_buffer", (PyCapsule_Destructor)&decref_capsule);
        if (PyArray_SetBaseObject((PyArrayObject*)arr, capsule) == -1) {
            Py_DECREF(arr);
            Py_XDECREF(input);
            return NULL;
        } 

    }else{
        std::cout << "not valid" << std::endl;
        memview = PyMemoryView_GetContiguous(input, PyBUF_WRITE, 'C');
        std::cout << "test" << std::endl;
        if(!memview) return NULL;
        PyObject* err = PyErr_Occurred();
        if (err) {
            PyErr_Clear();
            return NULL;
        }
        std::cout << "test" << std::endl;
        Py_buffer* buffer = NULL;
        if(PyMemoryView_Check(memview)) {
            buffer = PyMemoryView_GET_BUFFER(memview);
            // PyObject_GetBuffer(input, buffer, PyBUF_ANY_CONTIGUOUS);
        }
        if(!buffer) {
            Py_XDECREF(buffer);
            Py_XDECREF(memview);
            return NULL;
        }
        std::cout << "test" << std::endl;
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
            // PyObject* capsule = PyCapsule_New(input, "cpp_buffer", (PyCapsule_Destructor)&decref_capsule);
            // if (PyArray_SetBaseObject((PyArrayObject*)arr, capsule) == -1) {
            //     Py_DECREF(arr);
            //     Py_XDECREF(buffer);
            //     Py_XDECREF(memview);
            //     Py_XDECREF(input);
            //     return NULL;
            // } 
        }
    }
    
    return arr ? arr : NULL;
}

// static PyObject* cpp2np_array(PyObject* self, PyObject* args){
//     PyObject *arr;

//     int shape = 1;
//     int dims[] = {3};
//     auto *data = new std::array<int, 3>({1, 2, 3});
    
//     // arr = PyArray_SimpleNewFromData(shape, dims, NPY_DOUBLE, (void *)data);

//     PyObject *capsule = PyCapsule_New(data, "cpointer", (PyCapsule_Destructor)&free_wrap);
//     // PyArray_SetBaseObject((PyArrayObject *) arr, capsule);
    
//     return capsule;
// }

static char cpp2np_docs[] = {
    "C Numpy extension.\n"
};

static PyMethodDef cpp2np_funcs[] = {
    {"hello", (PyCFunction)hello, METH_VARARGS, "Hello World"},
    {"array", (PyCFunction)cpp2np_array, METH_VARARGS | METH_KEYWORDS, "create test array from c++ memory"},
    {"arrayCapsule", (PyCFunction)cpp2np_arrayCapsule, METH_VARARGS | METH_KEYWORDS, "create test array from c++ memory wrapped in pycapsule"},
    {"wrap", (PyCFunction)cpp2np_wrap, METH_VARARGS | METH_KEYWORDS, "create numpy array from pointer"},
    {"wrapFromCapsule", (PyCFunction)cpp2np_wrapFromCapsule, METH_VARARGS | METH_KEYWORDS, "create numpy array from pycapsule"},
    {"wrap2", (PyCFunction)cpp2np_wrap2, METH_VARARGS | METH_KEYWORDS, "create numpy array from memory view or capsule"},   
    {"getDescr", (PyCFunction)cpp2np_getDescr, METH_VARARGS | METH_KEYWORDS, "return array interface as dict"},
    {"free", (PyCFunction)cpp2np_free, METH_VARARGS | METH_KEYWORDS, "free pointer buffer"},
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
