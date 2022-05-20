#!/bin/bash

if [[ $1 != 'final' ]]
then
    twine upload --repository-url https://test.pypi.org/legacy/ dist/*
else
    echo 'twine upload dist/*'
fi
