from .operations import Placeholder
from .operations import Variable
from .operations import Constant

########################################
from .operations import AssignFrom
from .operations import Add
from .operations import Subtract
from .operations import Multiply
from .operations import Divide
from .operations import Reciprocal
from .operations import Sum
from .operations import Mean
from .operations import MatMul
from .operations import Negative
#######################################
from .operations import Square
from .operations import Sqrt
from .operations import Cube
from .operations import Cbrt
from .operations import Power
from .operations import Exp
from .operations import Log
from .operations import Sigmoid
from .operations import Sinh
from .operations import Arcsinh
from .operations import Cosh
from .operations import Arccosh
from .operations import Tanh
from .operations import Arctanh
from .operations import Relu
from .operations import LeakyRelu
from .operations import Swish
from .operations import Softplus
from .operations import Sin
from .operations import Arcsin
from .operations import Cos
from .operations import Arccos
from .operations import Tan
from .operations import Arctan
from .operations import Absolute
#########################################
from .operations import Ravel
from .operations import Reshape
from .operations import Maximum
from .operations import Minimum
from .operations import Max
from .operations import Min
from .operations import Where
from .operations import Transpose
from .operations import ZeroPad
from .operations import EdgePad
from .operations import Normalization
from .operations import Getitem
from .operations import Concatenate
from .operations import ExpandDims
from .operations import Flip
from .operations import Stack
from .operations import Swapaxes
from .operations import Moveaxis
from .operations import Rollaxis
from .operations import Squeeze
from .operations import Conv1D
from .operations import MaxPooling1D
from .operations import MeanPooling1D
from .operations import Conv2D
from .operations import MaxPooling2D
from .operations import MeanPooling2D
from .operations import Conv3D
from .operations import MaxPooling3D
from .operations import MeanPooling3D
#########################################
from .operations import Branch
#########################################
from .operations import Argmax
from .operations import Argmin
from .operations import Greater
from .operations import GreaterEqual
from .operations import Less
from .operations import LessEqual
from .operations import Equal
from .operations import NotEqual
from .operations import Any
from .operations import All
#########################################
from .operations import Executor
from .operations import Gradient
from .backend import backend as np

def as_ndarray(x):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    return x

def is_placeholer(x):
    return isinstance(x, Placeholder)

def is_constant(x):
    return isinstance(x, Constant)

def is_variable(x):
    return isinstance(x, Variable)

from .operations import Operation
def is_operation(x):
    return isinstance(x, Operation)

from .operations import ArithmeticalOperation
def is_arithmetical_operation(x):
    return isinstance(x, ArithmeticalOperation)

from .operations import LogicalOperation
def is_logical_operation(x):
    return isinstance(x, LogicalOperation)

def is_light(x):
    return is_placeholer(x) or is_constant(x) or is_variable(x) or is_operation(x)


def placeholder():
    return Placeholder()

def variable(initial_value):
    return Variable(as_ndarray(initial_value))

def constant(value):
    return Constant(as_ndarray(value))

def assign_from(x):
    if not is_light(x):
        raise TypeError('x is not light type')
    return AssignFrom(x)

def add(x, y):
    if not is_light(x):
        raise TypeError('x is not light type')
    if not is_light(y):
        raise TypeError('y is not light type')
    return Add(x, y)

def subtract(x, y):
    if not is_light(x):
        raise TypeError('x is not light type')
    if not is_light(y):
        raise TypeError('y is not light type')
    return Subtract(x, y)

def multiply(x, y):
    if not is_light(x):
        raise TypeError('x is not light type')
    if not is_light(y):
        raise TypeError('y is not light type')
    return Multiply(x, y)

def divide(x, y):
    if not is_light(x):
        raise TypeError('x is not light type')
    if not is_light(y):
        raise TypeError('y is not light type')
    return Divide(x, y)

def reciprocal(x):
    if not is_light(x):
        raise TypeError('x is not light type')
    return Reciprocal(x)

def sum(x, axis=None, keepdims=False):
    if not is_light(x):
        raise TypeError('x is not light type')
    return Sum(x, axis=axis, keepdims=keepdims)

def mean(x, axis=None, keepdims=False):
    if not is_light(x):
        raise TypeError('x is not light type')
    return Mean(x, axis=axis, keepdims=keepdims)

def matmul(x, y):
    if not is_light(x):
        raise TypeError('x is not light type')
    if not is_light(y):
        raise TypeError('y is not light type')
    return MatMul(x, y)

def negative(x):
    if not is_light(x):
        raise TypeError('x is not light type')
    return Negative(x)

def square(x):
    if not is_light(x):
        raise TypeError('x is not light type')
    return Square(x)

def sqrt(x):
    if not is_light(x):
        raise TypeError('x is not light type')
    return Sqrt(x)

def cube(x):
    if not is_light(x):
        raise TypeError('x is not light type')
    return Cube(x)

def cbrt(x):
    if not is_light(x):
        raise TypeError('x is not light type')
    return Cbrt(x)

def power(x, y):
    if not is_light(x):
        raise TypeError('x is not light type')
    if not is_light(y):
        raise TypeError('y is not light type')
    return Power(x, y)

def exp(x):
    if not is_light(x):
        raise TypeError('x is not light type')
    return Exp(x)

def log(x):
    if not is_light(x):
        raise TypeError('x is not light type')
    return Log(x)

def sigmoid(x):
    if not is_light(x):
        raise TypeError('x is not light type')
    return Sigmoid(x)

def sinh(x):
    if not is_light(x):
        raise TypeError('x is not light type')
    return Sinh(x)

def arcsinh(x):
    if not is_light(x):
        raise TypeError('x is not light type')
    return Arcsinh(x)

def cosh(x):
    if not is_light(x):
        raise TypeError('x is not light type')
    return Cosh(x)

def arccosh(x):
    if not is_light(x):
        raise TypeError('x is not light type')
    return Arccosh(x)

def tanh(x):
    if not is_light(x):
        raise TypeError('x is not light type')
    return Tanh(x)

def arctanh(x):
    if not is_light(x):
        raise TypeError('x is not light type')
    return Arctanh(x)

def relu(x):
    if not is_light(x):
        raise TypeError('x is not light type')
    return Relu(x)

def leaky_relu(x, leak=0.2):
    if not is_light(x):
        raise TypeError('x is not light type')
    return LeakyRelu(x, leak)

def swish(x, beta=1.0):
    if not is_light(x):
        raise TypeError('x is not light type')
    return Swish(x, beta)

def softplus(x):
    if not is_light(x):
        raise TypeError('x is not light type')
    return Softplus(x)

def sin(x):
    if not is_light(x):
        raise TypeError('x is not light type')
    return Sin(x)

def arcsin(x):
    if not is_light(x):
        raise TypeError('x is not light type')
    return Arcsin(x)

def cos(x):
    if not is_light(x):
        raise TypeError('x is not light type')
    return Cos(x)

def arccos(x):
    if not is_light(x):
        raise TypeError('x is not light type')
    return Arccos(x)

def tan(x):
    if not is_light(x):
        raise TypeError('x is not light type')
    return Tan(x)

def arctan(x):
    if not is_light(x):
        raise TypeError('x is not light type')
    return Arctan(x)

def abs(x):
    if not is_light(x):
        raise TypeError('x is not light type')
    return Absolute(x)

def ravel(x):
    if not is_light(x):
        raise TypeError('x is not light type')
    return Ravel(x)

def reshape(x, new_shape):
    if not is_light(x):
        raise TypeError('x is not light type')
    return Reshape(x, new_shape)

def maximum(x, y):
    if not is_light(x):
        raise TypeError('x is not light type')
    if not is_light(y):
        raise TypeError('y is not light type')
    return Maximum(x, y)

def minimum(x, y):
    if not is_light(x):
        raise TypeError('x is not light type')
    if not is_light(y):
        raise TypeError('y is not light type')
    return Minimum(x, y)

def max(x, axis=None, keepdims=False):
    if not is_light(x):
        raise TypeError('x is not light type')
    return Max(x, axis=axis, keepdims=keepdims)

def min(x, axis=None, keepdims=False):
    if not is_light(x):
        raise TypeError('x is not light type')
    return Min(x, axis=axis, keepdims=keepdims)

def where(cond, x, y):
    if not is_light(cond):
        raise TypeError('cond is not light type')
    if not is_light(x):
        raise TypeError('x is not light type')
    if not is_light(y):
        raise TypeError('y is not light type')
    return Where(cond, x, y)

def transpose(x, axes=None):
    if not is_light(x):
        raise TypeError('x is not light type')
    return Transpose(x, axes=axes)

#pad_width必须完整,不然会计算错误
def zero_pad(x, pad_width):
    if not is_light(x):
        raise TypeError('x is not light type')
    return ZeroPad(x, pad_width)

def edge_pad(x, pad_width):
    if not is_light(x):
        raise TypeError('x is not light type')
    return EdgePad(x, pad_width)

def normalization(cond, x, mean, std, momentum, eps):
    if not is_light(cond):
        raise TypeError('cond is not light type')
    if not is_light(x):
        raise TypeError('x is not light type')
    if not is_light(mean):
        raise TypeError('mean is not light type')
    if not is_light(std):
        raise TypeError('std is not light type')
    if not is_light(momentum):
        raise TypeError('momentum is not light type')
    if not is_light(eps):
        raise TypeError('eps is not light type')
    return Normalization(cond, x, mean, std, momentum, eps)

def getitem(x, key):
    if not is_light(x):
        raise TypeError('x is not light type')
    return Getitem(x, key)

def concatenate(lists, axis=0):
    return Concatenate(lists, axis)

def expand_dims(x, axis):
    if not is_light(x):
        raise TypeError('x is not light type')
    return ExpandDims(x, axis=axis)

def flip(x, axis=None):
    if not is_light(x):
        raise TypeError('x is not light type')
    return Flip(x, axis=axis)

def stack(x, axis=0):
    if not is_light(x):
        raise TypeError('x is not light type')
    return Stack(x, axis=axis)

def swapaxes(x, axis1, axis2):
    if not is_light(x):
        raise TypeError('x is not light type')
    return Swapaxes(x, axis1, axis2)

def moveaxis(x, source, destionation):
    if not is_light(x):
        raise TypeError('x is not light type')
    return Moveaxis(x, source, destionation)

def rollaxis(x, axis, start=0):
    if not is_light(x):
        raise TypeError('x is not light type')
    return Rollaxis(x, axis, start)

def squeeze(x, axis=None):
    if not is_light(x):
        raise TypeError('x is not light type')
    return Squeeze(x, axis)

def conv1d(x, kernel, stride):
    if not is_light(x):
        raise TypeError('x is not light type')
    if not is_light(kernel):
        raise TypeError('kernel is not light type')
    return Conv1D(x, kernel, stride)

def max_pooling1d(x, p_w, stride):
    if not is_light(x):
        raise TypeError('x is not light type')
    return MaxPooling1D(x, p_w, stride)

def mean_pooling1d(x, p_w, stride):
    if not is_light(x):
        raise TypeError('x is not light type')
    return MeanPooling1D(x, p_w, stride)

def conv2d(x, kernel, stride_h, stride_w):
    if not is_light(x):
        raise TypeError('x is not light type')
    if not is_light(kernel):
        raise TypeError('kernel is not light type')
    return Conv2D(x, kernel, (stride_h, stride_w))

def max_pooling2d(x, p_h, p_w, stride_h, stride_w):
    if not is_light(x):
        raise TypeError('x is not light type')
    return MaxPooling2D(x, (p_h, p_w), (stride_h, stride_w))

def mean_pooling2d(x, p_h, p_w, stride_h, stride_w):
    if not is_light(x):
        raise TypeError('x is not light type')
    return MeanPooling2D(x, (p_h, p_w), (stride_h, stride_w))

def conv3d(x, kernel, stride_d, stride_h, stride_w):
    if not is_light(x):
        raise TypeError('x is not light type')
    if not is_light(kernel):
        raise TypeError('kernel is not light type')
    return Conv3D(x, kernel, (stride_d, stride_h, stride_w))

def max_pooling3d(x, p_d, p_h, p_w, stride_d, stride_h, stride_w):
    if not is_light(x):
        raise TypeError('x is not light type')
    return MaxPooling3D(x, (p_d, p_h, p_w), (stride_d, stride_h, stride_w))

def mean_pooling3d(x, p_d, p_h, p_w, stride_d, stride_h, stride_w):
    if not is_light(x):
        raise TypeError('x is not light type')
    return MeanPooling3D(x, (p_d, p_h, p_w), (stride_d, stride_h, stride_w))

###########################################################################
def branch(cond, f0, f1, *nodes):
    if not is_light(cond):
        raise TypeError('cond is not light type')
    for n in nodes:
        if not is_light(n):
            raise TypeError('an node in nodes is not light type')
    return Branch(cond, f0, f1, *nodes)   

###########################################################################
def argmax(x, axis=None):
    if not is_light(x):
        raise TypeError('x is not light type')
    return Argmax(x, axis)

def argmin(x, axis=None):
    if not is_light(x):
        raise TypeError('x is not light type')
    return Argmin(x, axis)

def greater(x, y):
    if not is_light(x):
        raise TypeError('x is not light type')
    if not is_light(y):
        raise TypeError('y is not light type')
    return Greater(x, y)

def greater_equal(x, y):
    if not is_light(x):
        raise TypeError('x is not light type')
    if not is_light(y):
        raise TypeError('y is not light type')
    return GreaterEqual(x, y)

def less(x, y):
    if not is_light(x):
        raise TypeError('x is not light type')
    if not is_light(y):
        raise TypeError('y is not light type')
    return Less(x, y)

def less_equal(x, y):
    if not is_light(x):
        raise TypeError('x is not light type')
    if not is_light(y):
        raise TypeError('y is not light type')
    return LessEqual(x, y)

def equal(x, y):
    if not is_light(x):
        raise TypeError('x is not light type')
    if not is_light(y):
        raise TypeError('y is not light type')
    return Equal(x, y)

def not_equal(x, y):
    if not is_light(x):
        raise TypeError('x is not light type')
    if not is_light(y):
        raise TypeError('y is not light type')
    return NotEqual(x, y)

def any(x):
    if not is_light(x):
        raise TypeError('x is not light type')
    return Any(x)

def all(x):
    if not is_light(x):
        raise TypeError('x is not light type')
    return All(x)

###########################################################################
def executor(*objectives):
    for n in objectives:
        if not is_light(n):
            raise TypeError('an node in objectives is not light type')
    return Executor(objectives)

def gradient(objective, *variables):
    if not is_light(objective):
        raise TypeError('objective is not light type')
    for n in variables:
        if not is_light(n):
            raise TypeError('an node in variables is not light type')
    return Gradient(objective, variables)

###########################################################################