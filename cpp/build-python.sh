#!/bin/bash

python3 -m build
pip install --force-reinstall dist/*.whl
