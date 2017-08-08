#!/usr/bin/env python3

"""
This is a script to build the input specification file for julian's trained
networks.
"""

import json
import h5py
import numpy
from argparse import ArgumentParser
from itertools import chain, cycle, product
from math import ceil

def get_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('architecture_file')
    parser.add_argument('variable_names_file')
    parser.add_argument('-m', '--mean', required=True)
    parser.add_argument('-s', '--standard-deviation', required=True)
    return parser.parse_args()

def run():
    args = get_args()
    with open(args.architecture_file,'r') as arch_file:
        arch = json.load(arch_file)
        n_inputs = arch['config'][0]['config']['batch_input_shape'][1]
    with open(args.variable_names_file,'r') as vars_file:
        input_names = json.load(vars_file)

    mean = numpy.load(args.mean)
    std = numpy.load(args.standard_deviation)
    assert std.size == mean.size
    assert mean.size == n_inputs

    head_vars = input_names.get('header',[])
    repeat_vars = input_names.get('repeat', [])
    n_repeat_vars = len(repeat_vars)
    n_repeats = (n_inputs - len(head_vars)) / n_repeat_vars
    print(f'n_inputs: {n_inputs}, repeating: {n_repeat_vars}')
    print(f'number of repeats: {n_repeats}')

    def repeat_generator():
        for num, var in product(range(ceil(n_repeats)), repeat_vars):
            yield f'track_{num}_{var}'

    name_generator = chain(head_vars, repeat_generator())

    for num, (vname, mean, std) in enumerate(zip(name_generator, mean, std)):
        print(num, vname)
    # assert n_repeats * len(head_vars) + len(head_vars) == n_inputs

if __name__ == '__main__':
    run()
