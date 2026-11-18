#!/bin/bash

SCRIPT=`realpath $BASH_SOURCE`
SCRIPTPATH=`dirname $SCRIPT`

# Check whether uv is available
if ! command -v uv >/dev/null 2>&1
then
    echo "uv is required to set up the development environment." >&2
    echo "See https://docs.astral.sh/uv/getting-started/installation/." >&2
    return 1 2>/dev/null || exit 1
fi

# Create or update the virtual environment in .venv and install the project
# with all optional dependencies (extra arguments are passed to `uv sync`,
# e.g. `. setup.sh -p 3.12` to select a Python version)
if ! (cd "$SCRIPTPATH" && uv sync --all-extras "$@")
then
    return 1 2>/dev/null || exit 1
fi

# Activate the environment
. "$SCRIPTPATH/.venv/bin/activate" && echo "Development environment ready. Activated .venv."
