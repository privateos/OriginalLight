import numpy
#import cupy
backend = numpy

def set_cupy_as_backend():
    global backend
    backend = cupy

def set_numpy_as_backend():
    global backend
    backend = numpy

def set_backend(new_backend):
    global backend
    backend = new_backend
