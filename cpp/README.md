# Interactive Learning Package (C++/Python)

This python package allows training a SVM from opencv in C++.
It uses ANN Indexes to retrieve the top k suggestions based on the trained SVM.

## initialiaze()
Calling this function initializes the chosen index (HNSW, IVF, ANNOY) with the provided dataset.
There are several options related to each index.

## train()
Trains the SVM classifier based on the positive and negative items provided.

## suggest()
Gets the top k suggestions from the chosen index.

## reset_session()
Discards the current classifier and resets any session related variables.

## terminate()
Frees up all used memory from the IL session, index, and classifier.