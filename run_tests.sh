#!/bin/bash

echo "Running tests..." &&
coverage run --source 'remu' tests.py $@ &&
echo &&
echo "Coverage:" &&
echo &&
coverage report
