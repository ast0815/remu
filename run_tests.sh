#!/bin/bash

echo "Running tests..." &&
coverage run --source '.' --omit 'ENV/*' tests.py &&
echo &&
echo "Coverage:" &&
echo &&
coverage report
