#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "core_functions.h"

// Binding for add function
static PyObject* py_add(PyObject* self, PyObject* args) {
    float a, b;
    if (!PyArg_ParseTuple(args, "ff", &a, &b)) {
        return NULL;
    }
    return PyFloat_FromDouble(add(a, b));
}

// Binding for multiply function
static PyObject* py_multiply(PyObject* self, PyObject* args) {
    float a, b;
    if (!PyArg_ParseTuple(args, "ff", &a, &b)) {
        return NULL;
    }
    return PyFloat_FromDouble(multiply(a, b));
}

// Binding for print_hello function
static PyObject* py_print_hello(PyObject* self, PyObject* args) {
    print_hello();
    Py_RETURN_NONE;
}

// Binding for print_goodbye function
static PyObject* py_print_goodbye(PyObject* self, PyObject* args) {
    print_goodbye();
    Py_RETURN_NONE;
}

// Define module methods
static PyMethodDef Methods[] = {
    {"add", py_add, METH_VARARGS, "Add two numbers"},
    {"multiply", py_multiply, METH_VARARGS, "Multiply two numbers"},
    {"print_hello", py_print_hello, METH_VARARGS, "Print Hello World"},
    {"print_goodbye", py_print_goodbye, METH_VARARGS, "Print Goodbye"},
    {NULL, NULL, 0, NULL}
};

// Define module
static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "my_library",  // Module name
    NULL,  // No documentation
    -1,
    Methods
};

// Initialize module
PyMODINIT_FUNC PyInit_my_library(void) {
    return PyModule_Create(&module);
}
