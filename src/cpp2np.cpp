/// \file cpp2np.cpp

#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#include "numpy/arrayobject.h"
#include <string>
#include <array>
#include <iostream>
#include <memory>
 
static PyObject* hello(PyObject* self, PyObject* args){
    std::string s = "Hello Python, this is C++!";
    return Py_BuildValue("s", s.c_str());
};

/// Define a PyCapsule_Destructor for wrapper. merely serves as debug information
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

/**
 *  Creates a numpy array from a pointer to a c style array. The data area of the original array needs
 *  to be a contiguous block of memory
 * 
 *  \param pointer The pointer to the data as long int value
 *  \param shape A python tuple describing the shape of the c array
 *  \param dtype Keyword arg. A numpy dtype object indicating the data type of the array. Defaults to 'int32'
 *  \param free_mem_on_del Keyword arg. If True numpy array will free the data after its deletion. Defaults to False
 * 
 *  \return The numpy array if successful, otherwise NULL
 */
static PyObject* cpp2np_wrap(PyObject* self, PyObject* args, PyObject* kwargs){
    npy_intp ptr;  // npy_intp => intptr_t => unsigned long it
    npy_int free_mem_on_del=0;
    PyObject* arr = NULL;
    PyObject* in_shape = NULL;
    PyArray_Descr* dtype = NULL;
    
    std::string keys[] = {"input", "shape", "dtype", "free_mem_on_del"};
    static char* kwlist[] = {keys[0].data(), keys[1].data(), keys[2].data(), keys[3].data(), NULL};
    if(!PyArg_ParseTupleAndKeywords(args, kwargs, "lO|Oi", kwlist, &ptr, &in_shape, &dtype, &free_mem_on_del)){
        PyErr_BadArgument();
        return NULL;
    }
    if(dtype && !PyArray_DescrCheck(dtype)) {
        PyErr_SetString(PyExc_TypeError, "kwarg is not valid dtype object");
        return NULL;
    }
    if(!PyTuple_Check(in_shape)) {
        PyErr_SetString(PyExc_TypeError, "shape arg needs to be tuple");
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
    arr = PyArray_SimpleNewFromData(ndim, shape, dtype ? dtype->type_num : NPY_INT32, buf);

    // free memory by binding destructor capsule to numpy array
    // warning: this won't take into account the reference count on the buffer
    if(free_mem_on_del) {
        PyObject* capsule = PyCapsule_New(buf, "buffer_capsule", (PyCapsule_Destructor) &free_capsule);
        if (PyArray_SetBaseObject((PyArrayObject*)arr, capsule) == -1) {
            PyErr_SetString(PyExc_Exception, "failed to set array as base object to data");
            Py_DECREF(arr);
            return NULL;
        }
    }
    return arr;
};

/**
 *  Describes the data of a given numpy array
 *  
 *  \return A dict object containing the pointer, number of dimensions, shape and
 *      data type of the numpy array
 */
static PyObject* cpp2np_descr(PyObject* self, PyObject* args, PyObject* kwargs){
    PyObject* input = NULL;
    PyArrayObject* arr = NULL;
    PyObject* ret = NULL;
    
    std::string keys[] = {"input"};
    static char* kwlist[] = {keys[0].data(), NULL};
    if(!PyArg_ParseTupleAndKeywords(args, kwargs, "O", kwlist, &input)){
        PyErr_BadArgument();
        return NULL;
    }

    if(PyArray_Check(input)) {
        arr = (PyArrayObject*) input;
    } else {
        PyErr_SetString(PyExc_TypeError, "arg is not valid numpy array");
        return NULL;
    }
    
    // prepare return dict
    ret = PyDict_New();
    
    // first try to find array interface info
    PyObject* interf = NULL;
    if(PyObject_HasAttr(input, PyUnicode_FromString("__array_interface__"))) {
        interf = PyObject_GetAttr(input, PyUnicode_FromString("__array_interface__"));
        // return interf;
    }

    // get data pointer
    npy_intp data = (npy_intp) PyArray_DATA(arr);
    if(data){
        PyDict_SetItem(ret, PyUnicode_FromString("data"), PyLong_FromLong(data));
    }

    // get number of dimensions
    Py_ssize_t ndim = PyArray_NDIM(arr);
    if(ndim) {
        PyDict_SetItem(ret, PyUnicode_FromString("ndim"), PyLong_FromLong(ndim));
    }

    // try to get shape from array interface
    if(interf) {
        PyObject* shape = PyDict_GetItem(interf, PyUnicode_FromString("shape"));
        if(shape) {
            PyDict_SetItem(ret, PyUnicode_FromString("shape"), shape);
        }
    }
    // if interface not found, determine from API
    if(!interf) {
        npy_intp* shape = PyArray_SHAPE(arr);
        if(shape) {
            PyObject* shape_tuple = PyTuple_New(ndim);
            for(int i=0; i<ndim; ++i) {
                PyTuple_SET_ITEM(shape_tuple, i, PyLong_FromLong(shape[i]));
            }
            PyDict_SetItem(ret, PyUnicode_FromString("shape"), shape_tuple);
        }
    }

    // get typestr or typenum
    if(interf) {
        PyObject* typestr = PyDict_GetItem(interf, PyUnicode_FromString("typestr"));
        PyDict_SetItem(ret, PyUnicode_FromString("typestr"), typestr);
    } else {
        PyArray_Descr* dtype = PyArray_DTYPE(arr);
        if(PyArray_DescrCheck(dtype)) {
            PyDict_SetItem(ret, PyUnicode_FromString("typenum"), PyLong_FromLong(dtype->type_num));
        }
    }

    return ret;
};

/** Frees the memory to a given pointer
 * 
 *  \param pointer The pointer as long int value
 *  \return True if successful, otherwise False
 */
static PyObject* cpp2np_free(PyObject* self, PyObject* args, PyObject* kwargs){
    npy_intp ptr;
    std::string keys[] = {"pointer"};
    static char* kwlist[] = {keys[0].data(), NULL};
    if(!PyArg_ParseTupleAndKeywords(args, kwargs, "l", kwlist, &ptr)){
        PyErr_BadArgument();
        return Py_False;
    }

    void* buf = (void*) ptr;
    free(buf);
    
    return Py_True;
};

/// allocates a c++ array of 32 bit integer. \returns pointer and shape
static PyObject* cpp2np_c_arr_i4(PyObject* self, PyObject* args){
    auto* data = new std::array<std::array<int, 2>, 2>({{{1,2},{3,4}}});
    npy_intp ptr = (npy_intp) data;
    PyObject* ret = PyTuple_New(2);
    PyTuple_SET_ITEM(ret, 0, PyLong_FromLong(ptr));
    PyTuple_SET_ITEM(ret, 1, Py_BuildValue("(ii)", 2, 2));
    return ret;
};

/// allocates a c++ array of 64 bit integer. \returns pointer and shape  
static PyObject* cpp2np_c_arr_i8(PyObject* self, PyObject* args){
    auto* data = new std::array<std::array<long int, 2>, 2>({{{1,2},{3,4}}});
    npy_intp ptr = (npy_intp) data;
    PyObject* ret = PyTuple_New(2);
    PyTuple_SET_ITEM(ret, 0, PyLong_FromLong(ptr));
    PyTuple_SET_ITEM(ret, 1, Py_BuildValue("(ii)", 2, 2));
    return ret;
};

/// allocates a c++ array of doubles. \returns pointer and shape  
static PyObject* cpp2np_c_arr_f8(PyObject* self, PyObject* args){
    auto* data = new std::array<std::array<double, 2>, 2>({{{1.0, 2.0},{3.0, 4.0}}});
    npy_intp ptr = (npy_intp) data;
    PyObject* ret = PyTuple_New(2);
    PyTuple_SET_ITEM(ret, 0, PyLong_FromLong(ptr));
    PyTuple_SET_ITEM(ret, 1, Py_BuildValue("(ii)", 2, 2));
    return ret;
};

/// Returns a 8x8 pixel matrix used in the demo script \returns pointer and shape
static PyObject* cpp2np_c_arr_pixel(PyObject* self, PyObject* args){
    constexpr int size = 8;
    auto* data = new std::array<std::array<int16_t , size>, size>(
        {{{-18, 40, 48, 54, 42, 31, 6, 17},
         {38, 40, 36, 33, 37, 43, 31, 13},
         {18, -10,-4, -6, -9, 17, 34, 16},
         {-26, -94, -106, -103, -90, -17, 18, 31},
         {-21, -79, 2, 31, -126, -99, -11, 36},
         {-33, -57, 25, 79, -113, -98, -6, 22},
         {-16, -107, -128, -109, -128, -98, 4, 7},
         {35, 1, -45, -61, -59, -21, 11, 31}}}
    );

    npy_intp ptr = (npy_intp) data;
    PyObject* ret = PyTuple_New(2);
    PyTuple_SET_ITEM(ret, 0, PyLong_FromLong(ptr));
    PyTuple_SET_ITEM(ret, 1, Py_BuildValue("(ii)", size, size));

    std::cout << "Created pixel matrix with pointer: " << ptr << std::endl;

    return ret;
};

/// Prints an 8x8 matrix of int16 to standard output. \returns True if successful
static PyObject* cpp2np_print_arr(PyObject* self, PyObject* args, PyObject* kwargs){
    npy_intp ptr;
    std::string keys[] = {"pointer"};
    static char* kwlist[] = {keys[0].data(), NULL};
    if(!PyArg_ParseTupleAndKeywords(args, kwargs, "l", kwlist, &ptr)){
        PyErr_BadArgument();
        return Py_False;
    }

    constexpr int size = 8;
    typedef std::array<std::array<int16_t, size>, size> array_int16t_8x8;
    array_int16t_8x8 *matrix = (array_int16t_8x8 *) ptr;

    std::cout << "pointer address: " << ptr << std::endl;
    for (int i = 0; i < 8; ++i) {
        printf("[%4d %4d %4d %4d %4d %4d %4d %4d]\n", (*matrix)[i][0], (*matrix)[i][1], (*matrix)[i][2], (*matrix)[i][3], (*matrix)[i][4], (*matrix)[i][5], (*matrix)[i][6], (*matrix)[i][7]);
    }

    (*matrix)[0][0] = 255;  // to verify that memory is writable

    return Py_True;
};

/// stores docstring information of the module
static char cpp2np_docs[] = {
    "C Numpy extension for wrapping continuous C/C++ style arrays inside a numpy array using the same memory buffer.\n"
};

/// mapping of module functions and docstrings
static PyMethodDef cpp2np_funcs[] = {
    {"hello", (PyCFunction)hello, METH_VARARGS, "Hello World"},
    {"c_arr_i4", (PyCFunction)cpp2np_c_arr_i4, METH_VARARGS | METH_KEYWORDS, "Creates test array of int32 in C++"},
    {"c_arr_i8", (PyCFunction)cpp2np_c_arr_i8, METH_VARARGS | METH_KEYWORDS, "Creates test array of int64 in C++"},
    {"c_arr_f8", (PyCFunction)cpp2np_c_arr_f8, METH_VARARGS | METH_KEYWORDS, "Creates test array of doubles in C++"},
    {"c_arr_pixel", (PyCFunction)cpp2np_c_arr_pixel, METH_VARARGS | METH_KEYWORDS, "Creates array of pixels for demo script"},
    {"print_arr", (PyCFunction)cpp2np_print_arr, METH_VARARGS | METH_KEYWORDS, "Prints an 8x8 matrix of int16 to standard output"},
    {"wrap", (PyCFunction)cpp2np_wrap, METH_VARARGS | METH_KEYWORDS, "Creates numpy array from pointer"},
    {"descr", (PyCFunction)cpp2np_descr, METH_VARARGS | METH_KEYWORDS, "Returns a dict describing the data the numpy array"},
    {"free", (PyCFunction)cpp2np_free, METH_VARARGS | METH_KEYWORDS, "Frees the memory the pointer is referencing"},
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
    import_array();  // imports numpy support
    return PyModule_Create(&cpp2npmodule);
};