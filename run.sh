#!/usr/bin/env bash

if [[ $- == *i* ]]; then
    echo "don't source"
    return 1
fi

set -eu

./make-julian-variables-file.py data/arch.json data/vars-from-julian.json -s data/tracks_mean_vector.npy -m data/tracks_std_vector.npy
