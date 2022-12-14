#!/bin/bash
set -e

echo "Running examples..." &&
cd docs/examples/
for D in ??/
do
    cd $D
    if [[ -f do_everything.sh ]]
    then
        echo $D
        bash do_everything.sh
    else
        echo Skipping $D
    fi
    cd ..
done
