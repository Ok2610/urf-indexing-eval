# urf-indexing-eval

User Relevance Feedback Evaluation for ANN Indexes

# C++/Python Package

Under the "cpp" folder is the source code for building the python "il" package. It relies on opencv, hdf5, python3.10 and faiss (plan to make faiss optional in the future). Match the include directories and library directories for these in cpp/setup.py.
Once the setup.py is configured run the cpp/build-python.sh script.

# Evaluation Protocol

The evaluation script for this project is python/eval_ann.py.
Examples of how to run this script can be found in python/run_eval.sh

To use this evaluation script with Exquisitor (eCP) install Exquisitor from:
https://github.com/ITU-DASYALab/Exquisitor