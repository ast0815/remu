#!/bin/bash

mkdir -p build
mkdir -p dist
rm -rf build/* dist/*

python setup.py sdist
python setup.py bdist_wheel --universal
