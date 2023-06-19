///
/// Python Interface for Interactive Learning
///

#if defined(_WIN32)
#include <python3.10/Python.h>
#elif defined(__linux__)
#include <Python.h>
#endif

#include "PyInteractiveLearning.h"

namespace il {
    PyInteractiveLearning* py_il;

    static PyObject* initialize_py([[maybe_unused]] PyObject* self, PyObject* args);
    static PyObject* train_py([[maybe_unused]] PyObject* self, PyObject* args);
    static PyObject* suggest_py([[maybe_unused]] PyObject* self, PyObject* args);
    static PyObject* reset_model_py([[maybe_unused]] PyObject* self, [[maybe_unused]] PyObject* args);
    static PyObject* terminate_py([[maybe_unused]] PyObject* self, [[maybe_unused]] PyObject* args);

    static PyMethodDef il_methods[] = {
        {"initialize", initialize_py, METH_VARARGS, "Initialize index and classifier"},
        {"train", train_py, METH_VARARGS, "Train the classifier (linear SVM)"},
        {"suggest", suggest_py, METH_VARARGS, "Get suggestions from current model"},
        {"reset_model", reset_model_py, METH_VARARGS, "Reset the model"},
        {"safe_close", terminate_py, METH_NOARGS, "Safely free up memory"},
    };

    static PyModuleDef il_definition = {
            PyModuleDef_HEAD_INIT,
            "il",
            "A Python module that executes Interactive Learning functions.",
            -1,
            il_methods
    };

    PyMODINIT_FUNC PyInit_il(void);
}