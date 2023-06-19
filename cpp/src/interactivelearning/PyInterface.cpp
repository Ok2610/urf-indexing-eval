#include "PyInterface.h"

using std::cout;
using std::endl;
using std::string;
using std::vector;

PyObject* il::initialize_py([[maybe_unused]] PyObject* self, PyObject* args) {

    if (PyTuple_GET_SIZE(args) < 4) {
        cout << "Argument 0: Chosen Index (0: HNSW, 1: ANNOY, 2: IVF)" << endl;
        cout << "Argument 1: Datapoint representation (0: float, 1: R64)" << endl;
        cout << "Argument 2: Index Path" << endl;
        cout << "Argument 3: Index Parameters (List of arguments)" << endl;
        Py_INCREF(Py_None);
        return Py_None;
    }
    int ann_indx, mtype;
    const char* indx_path;
    PyObject* params;
    if (!PyArg_ParseTuple(args, "iisO!", &ann_indx, &mtype, &indx_path, &PyList_Type, &params))
        return NULL;
    int metric = (int)PyLong_AsLong(PyList_GetItem(params,0));
    int dim = (int)PyLong_AsLong(PyList_GetItem(params,1));
    int search_par = (int)PyLong_AsLong(PyList_GetItem(params,2)); // Runtime search parameter for index

    cout << "index: " << ann_indx << endl;
    cout << "mtype: " << mtype << endl;
    cout << "index path: " << indx_path << endl;
    cout << "metric: " << metric << endl;
    cout << "dim: " << dim << endl;
    cout << "search_par: " << search_par << endl;

    cout << "Loading index..." << endl;
    if (mtype == 0) {
        py_il = new PyInteractiveLearningFloat(ann_indx, indx_path, metric, dim, search_par);
    } else if (mtype == 1) {
        py_il = new PyInteractiveLearningR64();
        PyErr_SetString(PyExc_RuntimeError, "R64 is not implemented yet");
        return NULL;
    } else {
        PyErr_SetString(PyExc_RuntimeError, "No valid datapoint representation chosen");
        return NULL;
    }

    cout << "Index loaded!" << endl;

    Py_INCREF(Py_None);
    return Py_None;
};


PyObject* il::train_py([[maybe_unused]] PyObject* self, PyObject* args) {
    vector<vector<float>> train_items = vector<vector<float>>();
    vector<float> train_labels = vector<float>();

    if (PyTuple_GET_SIZE(args) < 2) {
        cout << "Argument 0: Training Items. 2D List of tuples [[(featId, featVal),...], [(featId,featVal),...], ...]" << endl;
        cout << "Argument 1: List of labels for each item. 1.0 or -1.0." << endl;
        return Py_None;
    }
    /// TODO: Make a helper functions header file to convert list to vector and vice versa.
    PyObject* py_train_items = PyTuple_GetItem(args, 0);
    PyObject* py_train_labels = PyTuple_GetItem(args, 1);
    for (int i = 0; i < PyList_Size(py_train_items); i++) { // Items
        vector<float> item_vector = vector<float>(py_il->_dim, 0.0);
        PyObject* item = PyList_GetItem(py_train_items,i);
        for (int j = 0; j < PyList_Size(item); j++) { // Tuples
            int f_id = (int) PyLong_AsLong(PyTuple_GetItem(PyList_GetItem(item,j), 0));
            float f_val = (float) PyFloat_AsDouble(PyTuple_GetItem(PyList_GetItem(item,j), 1));
            item_vector[f_id] = f_val;
        }
        train_items.push_back(item_vector);
        train_labels.push_back((float)PyFloat_AsDouble(PyList_GetItem(py_train_labels,i)));
    }

    // Input Check
    if (train_items.empty()) {
        PyErr_SetString(PyExc_RuntimeError, "No train items provided!");
        return 0;
    }
    if (train_labels.empty()) {
        PyErr_SetString(PyExc_RuntimeError, "No train labels provided!");
        return 0;
    }
    if (train_labels.size() != train_items.size()) {
        PyErr_SetString(PyExc_RuntimeError, "Train items and labels mismatch!");
        return 0;
    }

    vector<float> weights = py_il->_classifier->train(train_items, train_labels);

    PyObject* hyperplane = PyList_New(weights.size());
    for (size_t i = 0; i < weights.size(); i++) {
        PyList_SetItem(hyperplane, i, PyFloat_FromDouble(weights[i]));
    }
    
    return hyperplane;
};


PyObject* il::suggest_py([[maybe_unused]] PyObject* self, PyObject* args) {

    if (PyTuple_GET_SIZE(args) == 0) {
        cout << "Argument 0: k items to return" << endl;
        cout << "Argument 1: (Optional) search parameter for index" << endl;
        Py_INCREF(Py_None);
        return Py_None;
    }

    // cout << "(C++) Getting suggestions" << endl;
    int k = (int)PyLong_AsLong(PyTuple_GetItem(args,0));
    vector<uint32_t> items;
    if (PyTuple_GET_SIZE(args) == 1) {
        items = py_il->get_suggestions(k, -1);
    } else {
        int search_par = (int)PyLong_AsLong(PyTuple_GetItem(args,1));
        items = py_il->get_suggestions(k, search_par);
    }

    // cout << "(C++) Preparing return list with " << items.size() << " items" << endl;
    PyObject* py_items = PyList_New(items.size());
    for (size_t i = 0; i < items.size(); i++) {
        PyList_SetItem(py_items, i, PyLong_FromLong(items[i]));
    }

    return py_items;
};


PyObject* il::reset_model_py([[maybe_unused]] PyObject* self, [[maybe_unused]] PyObject* args) {
    py_il->_classifier->reset_classifier();

    Py_INCREF(Py_None);
    return Py_None;
};


PyObject* il::terminate_py([[maybe_unused]] PyObject* self, [[maybe_unused]] PyObject* args) {
    delete py_il;

    Py_INCREF(Py_None);
    return Py_None;
};

PyMODINIT_FUNC il::PyInit_il(void) {
    Py_Initialize();
    return PyModule_Create(&il_definition);
}