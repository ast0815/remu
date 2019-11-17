#!/bin/bash

# Check whether the ENV exists
if [ -d ENV ]
then
    # Activate the environment
    echo "Activating virtual environment."
    . ENV/bin/activate
else
    # Create a new virtual environment
    virtualenv ENV

    # Activate the environment
    . ENV/bin/activate && (

        # Upgrade pip to the latest version (optional)
        pip install --upgrade pip

        # Install all required packages
        pip install -r requirements.txt
        pip install -r test-requirements.txt
        pip install -r documentation-requirements.txt

        # Install actual package
        pip install -e .

    )
fi
