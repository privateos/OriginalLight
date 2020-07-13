from .backend import backend as np

class Placeholder(object):
    def __init__(self):
        self.output_value = None
        self.output_nodes = []

class Variable(object):
    def __init__(self, initial_value): 
        self.output_value = initial_value
        self.output_nodes = []

    def compute_output(self):
        return self.output_value

class Constant(object):
    def __init__(self, value):
        self.output_value = value
        self.output_nodes = []

    def compute_output(self):
        return self.output_value




class Operation(object):
    pass

class ArithmeticalOperation(Operation):
    def __init__(self, *input_nodes):
        self.input_nodes = input_nodes
        self.output_nodes = []
        self.output_value = None
        for node in input_nodes:
            node.output_nodes.append(self)

    def compute_output(self):
        raise NotImplementedError

    def gradients_function(self):
        raise NotImplementedError

class LogicalOperation(Operation):
    def __init__(self, *input_nodes):
        self.input_nodes = input_nodes
        self.output_nodes = []
        self.output_value = None
        for node in input_nodes:
            node.output_nodes.append(self)

    def compute_output(self):
        raise NotImplementedError



"""
ArithmeticalOperation
"""
class AssignFrom(ArithmeticalOperation):
    def __init__(self, x):
        super(self.__class__, self).__init__(x)

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = x.output_value
        return self.output_value

    def gradients_function(self):
        def grad_x(grad):
            return grad
        return [grad_x]

#pass
class Add(ArithmeticalOperation):
    def __init__(self, x, y):
        super(self.__class__, self).__init__(x, y)

    def compute_output(self):
        x, y = self.input_nodes
        self.output_value = np.add(x.output_value, y.output_value)
        return self.output_value

    def gradients_function(self):
        def grad_x(grad):
            x = self.input_nodes[0].output_value
            grad_wrt_x = grad
            sum_times = np.ndim(grad_wrt_x) - np.ndim(x)
            for i in range(sum_times):
                grad_wrt_x = np.sum(grad_wrt_x, axis=0)
            for axis, size in enumerate(np.shape(x)):
                if size == 1:
                    grad_wrt_x = np.sum(grad_wrt_x, axis=axis, keepdims=True)
            return grad_wrt_x

        def grad_y(grad):
            y = self.input_nodes[1].output_value
            grad_wrt_y = grad
            sum_times = np.ndim(grad_wrt_y) - np.ndim(y)
            for i in range(sum_times):
                grad_wrt_y = np.sum(grad_wrt_y, axis=0)
            for axis, size in enumerate(np.shape(y)):
                if size == 1:
                    grad_wrt_y = np.sum(grad_wrt_y, axis=axis, keepdims=True)
            return grad_wrt_y
        
        return [grad_x, grad_y]

#pass        
class Subtract(ArithmeticalOperation):
    def __init__(self, x, y):
        super(self.__class__, self).__init__(x, y)

    def compute_output(self):
        x, y = self.input_nodes
        self.output_value = np.subtract(x.output_value, y.output_value)
        return self.output_value

    def gradients_function(self):
        def grad_x(grad):
            x = self.input_nodes[0].output_value
            grad_wrt_x = grad
            sum_times = np.ndim(grad_wrt_x) - np.ndim(x)
            for i in range(sum_times):
                grad_wrt_x = np.sum(grad_wrt_x, axis=0)
            for axis, size in enumerate(np.shape(x)):
                if size == 1:
                    grad_wrt_x = np.sum(grad_wrt_x, axis=axis, keepdims=True)
            return grad_wrt_x
        
        def grad_y(grad):
            y = self.input_nodes[1].output_value
            grad_wrt_y = np.negative(grad)
            sum_times = np.ndim(grad_wrt_y) - np.ndim(y)
            for i in range(sum_times):
                grad_wrt_y = np.sum(grad_wrt_y, axis=0)
            for axis, size in enumerate(np.shape(y)):
                if size == 1:
                    grad_wrt_y = np.sum(grad_wrt_y, axis=axis, keepdims=True)
            return grad_wrt_y

        return [grad_x, grad_y]

#pass
class Multiply(ArithmeticalOperation):
    def __init__(self, x, y):
        super(self.__class__, self).__init__(x, y)

    def compute_output(self):
        x, y = self.input_nodes
        self.output_value = np.multiply(x.output_value, y.output_value)
        return self.output_value

    def gradients_function(self):
        def grad_x(grad):
            x, y = [node.output_value for node in self.input_nodes]
            
            grad_wrt_x = np.multiply(grad, y)
            sum_times = np.ndim(grad_wrt_x) - np.ndim(x)
            for i in range(sum_times):
                grad_wrt_x = np.sum(grad_wrt_x, axis=0)
            for axis, size in enumerate(np.shape(x)):
                if size == 1:
                    grad_wrt_x = np.sum(grad_wrt_x, axis=axis, keepdims=True)
            return grad_wrt_x

        def grad_y(grad):
            x, y = [node.output_value for node in self.input_nodes]
            grad_wrt_y = np.multiply(grad, x)
            sum_times = np.ndim(grad_wrt_y) - np.ndim(y)
            for i in range(sum_times):
                grad_wrt_y = np.sum(grad_wrt_y, axis=0)
            for axis, size in enumerate(np.shape(y)):
                if size == 1:
                    grad_wrt_y = np.sum(grad_wrt_y, axis=axis, keepdims=True)
            return grad_wrt_y
        return [grad_x, grad_y]

#pass
class Divide(ArithmeticalOperation):
    def __init__(self, x, y):
        super(self.__class__, self).__init__(x, y)

    def compute_output(self):
        x, y = self.input_nodes
        self.output_value = np.divide(x.output_value, y.output_value)
        return self.output_value

    def gradients_function(self):
        def grad_x(grad):
            x, y = [node.output_value for node in self.input_nodes]
            
            grad_wrt_x = np.multiply(grad, np.divide(1.0, y))
            sum_times = np.ndim(grad_wrt_x) - np.ndim(x)
            for i in range(sum_times):
                grad_wrt_x = np.sum(grad_wrt_x, axis=0)
            for axis, size in enumerate(np.shape(x)):
                if size == 1:
                    grad_wrt_x = np.sum(grad_wrt_x, axis=axis, keepdims=True)
            return grad_wrt_x

        def grad_y(grad):
            x, y = [node.output_value for node in self.input_nodes]
     
            grad_wrt_y = np.negative(np.multiply(grad, np.divide(x, np.square(y))))
            sum_times = np.ndim(grad_wrt_y) - np.ndim(y)
            for i in range(sum_times):
                grad_wrt_y = np.sum(grad_wrt_y, axis=0)
            for axis, size in enumerate(np.shape(y)):
                if size == 1:
                    grad_wrt_y = np.sum(grad_wrt_y, axis=axis, keepdims=True)
            return grad_wrt_y

        return [grad_x, grad_y]       

#pass
class Reciprocal(ArithmeticalOperation):
    def __init__(self, x):
        super(self.__class__, self).__init__(x)

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = np.reciprocal(x.output_value)
        return self.output_value

    def gradients_function(self):
        def grad_x(grad):
            return np.multiply(grad, np.negative(np.square(self.output_value)))

        return [grad_x]       

#pass
class Sum(ArithmeticalOperation):
    def __init__(self, x, axis=None, keepdims=False):
        super(self.__class__, self).__init__(x)
        self.axis = axis
        self.keepdims = keepdims

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = np.sum(x.output_value, axis=self.axis, keepdims=self.keepdims)
        return self.output_value

    def gradients_function(self):
        def grad_x(grad):
            input_value = self.input_nodes[0].output_value
            output_shape = np.array(np.shape(input_value))
            output_shape[self.axis] = 1
            tile_scaling = np.shape(input_value)//output_shape
            return np.tile(grad, tile_scaling)
        
        return [grad_x]

#pass
class Mean(ArithmeticalOperation):
    def __init__(self, x, axis=None, keepdims=False):
        super(self.__class__, self).__init__(x)
        self.axis = axis
        self.keepdims = keepdims

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = np.mean(x.output_value, axis=self.axis, keepdims=self.keepdims)
        return self.output_value

    def gradients_function(self):
        def grad_x(grad):
            input_value = self.input_nodes[0].output_value
            output_shape = np.array(np.shape(input_value))
            d = np.prod(output_shape[self.axis])
            grad = np.divide(grad, d)
            output_shape[self.axis] = 1
            tile_scaling = np.shape(input_value)//output_shape
            return np.tile(grad, tile_scaling)
        
        return [grad_x]

#pass
class MatMul(ArithmeticalOperation):
    def __init__(self, x, y):
        super(self.__class__, self).__init__(x, y)

    def compute_output(self):
        x, y = self.input_nodes
        self.output_value = np.dot(x.output_value, y.output_value)
        return self.output_value
    
    def gradients_function(self):
        def grad_x(grad):
            y = self.input_nodes[1].output_value
            return np.dot(grad, np.transpose(y))

        def grad_y(grad):
            x = self.input_nodes[0].output_value
            return np.dot(np.transpose(x), grad)

        return[grad_x, grad_y]

#pass
class Negative(ArithmeticalOperation):
    def __init__(self, x):
        super(self.__class__, self).__init__(x)

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = np.negative(x.output_value)
        return self.output_value

    def gradients_function(self):
        def grad_x(grad):
            return np.negative(grad)
        return [grad_x]

#pass
class Square(ArithmeticalOperation):
    def __init__(self, x):
        super(self.__class__, self).__init__(x)

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = np.square(x.output_value)
        return self.output_value

    def gradients_function(self):
        def grad_x(grad):
            x, = self.input_nodes
            return np.multiply(grad, np.multiply(2.0, x.output_value))
        
        return [grad_x]

#pass
class Sqrt(ArithmeticalOperation):
    def __init__(self, x):
        super(self.__class__, self).__init__(x)

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = np.sqrt(x.output_value)
        return self.output_value

    def gradients_function(self):
        def grad_x(grad):
            return np.multiply(grad, np.multiply(0.5, np.divide(1.0, self.output_value)))
        
        return [grad_x]

#pass
class Cube(ArithmeticalOperation):
    def __init__(self, x):
        super(self.__class__, self).__init__(x)

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = np.power(x.output_value, 3.0)
        return self.output_value

    def gradients_function(self):
        def grad_x(grad):
            x, = self.input_nodes
            return np.multiply(grad, np.multiply(3.0, np.square(x.output_value)))
        
        return [grad_x]

#pass
class Cbrt(ArithmeticalOperation):
    def __init__(self, x):
        super(self.__class__, self).__init__(x)

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = np.cbrt(x.output_value)
        return self.output_value

    def gradients_function(self):
        def grad_x(grad):
            return np.multiply(grad, np.multiply(1.0/3.0, np.square(np.reciprocal(self.output_value))))
        
        return [grad_x]

#pass
class Power(ArithmeticalOperation):
    def __init__(self, x, y):
        super(self.__class__, self).__init__(x, y)

    def compute_output(self):
        x, y= self.input_nodes
        self.output_value = np.power(x.output_value, y.output_value)
        return self.output_value

    def gradients_function(self):
        def grad_x(grad):
            x, y = self.input_nodes
            grad = np.multiply(grad, np.multiply(y.output_value, np.power(x.output_value, np.subtract(y.output_value, 1.0))))
            
            for _ in range(np.ndim(grad) - np.ndim(x.output_value)):
                grad = np.sum(grad, axis=0)
            
            for i, dim in enumerate(np.shape(x.output_value)):
                if dim == 1:
                    grad = np.sum(grad, axis=i, keepdims=True)
            
            return grad
        
        def grad_y(grad):
            x, y = self.input_nodes
            grad = np.multiply(grad, np.multiply(np.log(x.output_value), self.output_value))
            
            for _ in range(np.ndim(grad) - np.ndim(y.output_value)):
                grad = np.sum(grad, axis=0)
            
            for i, dim in enumerate(np.shape(y.output_value)):
                if dim == 1:
                    grad = np.sum(grad, axis=i, keepdims=True)
            
            return grad

        return [grad_x, grad_y]

#pass
class Exp(ArithmeticalOperation):
    def __init__(self, x):
        super(self.__class__, self).__init__(x)

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = np.exp(x.output_value)
        return self.output_value
        
    def gradients_function(self):
        def grad_x(grad):
            return np.multiply(grad, self.output_value)
        return [grad_x]

#pass
class Log(ArithmeticalOperation):
    def __init__(self, x):
        super(self.__class__, self).__init__(x)

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = np.log(x.output_value)
        return self.output_value
        
    def gradients_function(self):
        def grad_x(grad):
            x = self.input_nodes[0].output_value
            return np.divide(grad, x)
        return [grad_x]

#pass
class Sigmoid(ArithmeticalOperation):
    def __init__(self, x):
        super(self.__class__, self).__init__(x)

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = np.divide(1.0, np.add(1.0, np.exp(np.negative(x.output_value))))
        return self.output_value

    def gradients_function(self):
        def grad_x(grad):
            return np.multiply(grad, np.multiply(self.output_value, np.subtract(1.0, self.output_value)))
        return [grad_x]

#pass
class Sinh(ArithmeticalOperation):
    def __init__(self, x):
        super(self.__class__, self).__init__(x)
        self.ex_2 = None
        self.e_x_2 = None

    def compute_output(self):
        x, = self.input_nodes
        ex = np.exp(x.output_value)
        self.ex_2 = np.multiply(ex, 0.5)
        self.e_x_2 = np.multiply(0.5, np.reciprocal(ex))
        self.output_value = np.subtract(self.ex_2, self.e_x_2)
        return self.output_value

    def gradients_function(self):
        def grad_x(grad):
            return np.multiply(grad, np.add(self.ex_2, self.e_x_2))
        return [grad_x]

#pass
class Arcsinh(ArithmeticalOperation):
    def __init__(self, x):
        super(self.__class__, self).__init__(x)
        self.sqrtx_1 = None

    def compute_output(self):
        x, = self.input_nodes
        self.sqrtx_1 = np.sqrt(np.add(1.0, np.square(x.output_value)))
        self.output_value = np.log(np.add(x.output_value, self.sqrtx_1))
        return self.output_value

    def gradients_function(self):
        def grad_x(grad):
            return np.multiply(grad, np.reciprocal(self.sqrtx_1))
        return [grad_x]

#pass
class Cosh(ArithmeticalOperation):
    def __init__(self, x):
        super(self.__class__, self).__init__(x)
        self.ex_2 = None
        self.e_x_2 = None

    def compute_output(self):
        x, = self.input_nodes
        ex = np.exp(x.output_value)
        self.ex_2 = np.multiply(0.5, ex)
        self.e_x_2 = np.multiply(0.5, np.reciprocal(ex))
        self.output_value = np.add(self.ex_2, self.e_x_2)
        return self.output_value

    def gradients_function(self):
        def grad_x(grad):
            return np.multiply(grad, np.subtract(self.ex_2, self.e_x_2))
        return [grad_x]

#pass
class Arccosh(ArithmeticalOperation):
    def __init__(self, x):
        super(self.__class__, self).__init__(x)
        self.sqrtx_1 = None

    def compute_output(self):
        x, = self.input_nodes
        self.sqrtx_1 = np.sqrt(np.subtract(np.square(x.output_value), 1.0))
        self.output_value = np.log(np.add(x.output_value, self.sqrtx_1))
        return self.output_value

    def gradients_function(self):
        def grad_x(grad):
            return np.multiply(grad, np.reciprocal(self.sqrtx_1))
        return [grad_x]

#pass
class Tanh(ArithmeticalOperation):
    def __init__(self, x):
        super(self.__class__, self).__init__(x)

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = np.tanh(x.output_value)
        return self.output_value
    
    def gradients_function(self):
        def grad_x(grad):
            return np.multiply(grad, np.subtract(1.0, np.square(self.output_value)))
        return [grad_x]

#pass
class Arctanh(ArithmeticalOperation):
    def __init__(self, x):
        super(self.__class__, self).__init__(x)

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = np.multiply(0.5, np.log(np.divide(np.add(1.0, x.output_value), np.subtract(1.0, x.output_value))))
        return self.output_value
    
    def gradients_function(self):
        def grad_x(grad):
            x, = self.input_nodes
            return np.multiply(grad, np.reciprocal(np.subtract(1.0, np.square(x.output_value))))
        return [grad_x]

#need to optimize this class
#pass
class Relu(ArithmeticalOperation):
    def __init__(self, x):
        super(self.__class__, self).__init__(x)

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = np.maximum(x.output_value, 0.0)
        return self.output_value

    
    def gradients_function(self):
        def grad_x(grad):
            grad = np.copy(grad)
            grad[np.equal(self.output_value, 0.0)] = 0.0
            return grad
        return [grad_x]

class LeakyRelu(ArithmeticalOperation):
    def __init__(self, x, leak=0.2):
        super(self.__class__, self).__init__(x)
        self.leak = leak
        self.lx = None
    
    def compute_output(self):
        x, = self.input_nodes
        self.lx = self.leak*x.output_value
        self.output_value = np.maximum(x.output_value, self.lx)
        return self.output_value

    def gradients_function(self):
        def grad_x(grad):
            x, = self.input_nodes
            cond = x.output_value > self.lx
            return grad*np.where(cond, 1.0, self.leak)
        return [grad_x]

class Elu(ArithmeticalOperation):
    def __init__(self, x, leak=0.2):
        super(self.__class__, self).__init__(x)
        self.leak = leak
        self.lx = None
    
    def compute_output(self):
        x, = self.input_nodes
        self.lx = self.leak*x.output_value
        self.output_value = np.maximum(x.output_value, self.lx)
        return self.output_value

    def gradients_function(self):
        def grad_x(grad):
            x, = self.input_nodes
            cond = x.output_value > self.lx
            return grad*np.where(cond, 1.0, self.leak)
        return [grad_x]

#pass
class Swish(ArithmeticalOperation):
    def __init__(self, x, beta=1.0):
        super(self.__class__, self).__init__(x)
        self.beta = beta
        self.bx = None # beta*x
        self.ebx = None#exp(-beta*x)
        self.ebx1 = None#1.0/(1 + exp(-beta*x))


    def compute_output(self):
        x, = self.input_nodes
        self.bx = self.beta*x.output_value
        self.ebx = np.exp(-self.bx)
        self.ebx1 = 1.0/(1.0 + self.ebx)
        self.output_value = x.output_value*self.ebx1
        return self.output_value
    
    def gradients_function(self):
        def grad_x(grad):
            return self.ebx1 + self.bx*self.ebx*np.square(self.ebx1)
        return [grad_x]

#pass
class Softplus(ArithmeticalOperation):
    def __init__(self, x):
        super(self.__class__, self).__init__(x)
        self.ex = None#exp(x)
        self.ex1 = None# 1 + exp(x)


    def compute_output(self):
        x, = self.input_nodes
        self.ex = np.exp(x.output_value)
        self.ex1 = 1.0 + self.ex
        self.output_value = np.log(self.ex1)
        return self.output_value
    
    def gradients_function(self):
        def grad_x(grad):
            return grad* self.ex/self.ex1
        return [grad_x]
        
#pass
class Sin(ArithmeticalOperation):
    def __init__(self, x):
        super(self.__class__, self).__init__(x)
    
    def compute_output(self):
        x, = self.input_nodes
        self.output_value = np.sin(x.output_value)
        return self.output_value

    def gradients_function(self):
        def grad_x(grad):
            x, = self.input_nodes
            return np.multiply(grad, np.cos(x.output_value))
        return [grad_x]

#pass
class Arcsin(ArithmeticalOperation):
    def __init__(self, x):
        super(self.__class__, self).__init__(x)
    
    def compute_output(self):
        x, = self.input_nodes
        self.output_value = np.arcsin(x.output_value)
        return self.output_value

    def gradients_function(self):
        def grad_x(grad):
            x, = self.input_nodes
            return np.multiply(grad, np.divide(1.0, np.sqrt(np.subtract(1.0, np.square(x.output_value)))))
        return [grad_x]

#pass
class Cos(ArithmeticalOperation):
    def __init__(self, x):
        super(self.__class__, self).__init__(x)
    
    def compute_output(self):
        x, = self.input_nodes
        self.output_value = np.cos(x.output_value)
        return self.output_value

    def gradients_function(self):
        def grad_x(grad):
            x, = self.input_nodes
            return np.multiply(grad, np.negative(np.sin(x.output_value)))
        return [grad_x]

#pass
class Arccos(ArithmeticalOperation):
    def __init__(self, x):
        super(self.__class__, self).__init__(x)
    
    def compute_output(self):
        x, = self.input_nodes
        self.output_value = np.arccos(x.output_value)
        return self.output_value

    def gradients_function(self):
        def grad_x(grad):
            x, = self.input_nodes
            return np.multiply(grad, np.negative(np.divide(1.0, np.sqrt(np.subtract(1.0, np.square(x.output_value))))))
        return [grad_x]

#pass
class Tan(ArithmeticalOperation):
    def __init__(self, x):
        super(self.__class__, self).__init__(x)
    
    def compute_output(self):
        x, = self.input_nodes
        self.output_value = np.tan(x.output_value)
        return self.output_value

    def gradients_function(self):
        def grad_x(grad):
            x, = self.input_nodes
            return np.multiply(grad, np.divide(1.0, np.square(np.cos(x.output_value))))
        return [grad_x]

#pass
class Arctan(ArithmeticalOperation):
    def __init__(self, x):
        super(self.__class__, self).__init__(x)
    
    def compute_output(self):
        x, = self.input_nodes
        self.output_value = np.arctan(x.output_value)
        return self.output_value

    def gradients_function(self):
        def grad_x(grad):
            x, = self.input_nodes
            return np.multiply(grad, np.divide(1.0, np.add(1.0, np.square(x.output_value))))
        return [grad_x]

#pass
class Absolute(ArithmeticalOperation):
    def __init__(self, x):
        super(self.__class__, self).__init__(x)
        self.cache = None

    def compute_output(self):
        x, = self.input_nodes
        self.cache = np.less(x.output_value, 0)#x.output_value < 0
        self.output_value = np.where(self.cache, np.negative(x.output_value), x.output_value)
        return self.output_value

    def gradients_function(self):
        def grad_x(grad):
            x, = self.input_nodes
            t = np.ones_like(x.output_value)
            t.__setitem__(self.cache, -1.0)
            return np.multiply(grad, t)#grad*t
        return [grad_x]

#pass
class Ravel(ArithmeticalOperation):
    def __init__(self, x):
        super(self.__class__, self).__init__(x)

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = np.ravel(x.output_value)
        return self.output_value

    def gradients_function(self):
        def grad_x(grad):
            x, = self.input_nodes
            return np.reshape(grad, np.shape(x.output_value))
        return [grad_x]

#pass
class Reshape(ArithmeticalOperation):
    def __init__(self, x, new_shape):
        super(self.__class__, self).__init__(x)
        self.new_shape = new_shape

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = np.reshape(x.output_value, self.new_shape)
        return self.output_value

    def gradients_function(self):
        def grad_x(grad):
            x, = self.input_nodes
            return np.reshape(grad, np.shape(x.output_value))
        return [grad_x]

#pass
class Maximum(ArithmeticalOperation):
    def __init__(self, x, y):
        super(self.__class__, self).__init__(x, y)
        self.cache = None

    def compute_output(self):
        x, y = self.input_nodes
        self.cache = np.greater(x.output_value, y.output_value)
        self.output_value = np.where(self.cache, x.output_value, y.output_value)
        return self.output_value

    def gradients_function(self):
        def grad_x(grad):
            x, y = self.input_nodes
            grad = np.where(self.cache, grad, 0.0)
            for _ in range(np.ndim(grad) - np.ndim(x.output_value)):
                grad = np.sum(grad, axis=0)
            for i, dim in enumerate(np.shape(x.output_value)):
                if dim == 1:
                    grad = np.sum(grad, axis=i, keepdims=True)
            return grad
        
        def grad_y(grad):
            x, y = self.input_nodes
            grad = np.where(self.cache, 0.0, grad)
            for _ in range(np.ndim(grad) - np.ndim(y.output_value)):
                grad = np.sum(grad, axis=0)
            for i, dim in enumerate(np.shape(y.output_value)):
                if dim == 1:
                    grad = np.sum(grad, axis=i, keepdims=True)
            return grad
        return [grad_x, grad_y]

#pass
class Minimum(ArithmeticalOperation):
    def __init__(self, x, y):
        super(self.__class__, self).__init__(x, y)
        self.cache = None

    def compute_output(self):
        x, y = self.input_nodes
        self.cache = np.less(x.output_value, y.output_value)
        self.output_value = np.where(self.cache, x.output_value, y.output_value)
        return self.output_value

    def gradients_function(self):
        def grad_x(grad):
            x, y = self.input_nodes
            grad = np.where(self.cache, grad, 0.0)

            for _ in range(np.ndim(grad) - np.ndim(x.output_value)):
                grad = np.sum(grad, axis=0)
            for i, dim in enumerate(np.shape(x.output_value)):
                if dim == 1:
                    grad = np.sum(grad, axis=i, keepdims=True)
            return grad

        def grad_y(grad):
            x, y = self.input_nodes
            grad = np.where(self.cache, 0.0, grad)

            for _ in range(np.ndim(grad) - np.ndim(y.output_value)):
                grad = np.sum(grad, axis=0)
            for i, dim in enumerate(np.shape(y.output_value)):
                if dim == 1:
                    grad = np.sum(grad, axis=i, keepdims=True)
            return grad

        return [grad_x, grad_y]

#pass
class Max(ArithmeticalOperation):
    def __init__(self, x, axis=None, keepdims=False):
        super(self.__class__, self).__init__(x)
        self.axis = axis
        self.keepdims = keepdims

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = np.max(x.output_value, axis=self.axis, keepdims=self.keepdims)
        return self.output_value

    def gradients_function(self):
        def grad_x(grad):
            x, = self.input_nodes
            if self.keepdims:
                return np.multiply(grad, np.equal(self.output_value, x.output_value))
            else:
                outshape = np.array(np.shape(x.output_value))
                outshape[self.axis] = 1
                t = np.shape(x.output_value)//outshape
                r = np.reshape(self.output_value, outshape)
                g = np.tile(r, t)
                grad = np.reshape(grad, outshape)
                eq = np.equal(g, x.output_value)
                return np.multiply(grad, eq)
        return [grad_x]

#pass
class Min(ArithmeticalOperation):
    def __init__(self, x, axis=None, keepdims=False):
        super(self.__class__, self).__init__(x)
        self.axis = axis
        self.keepdims = keepdims

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = np.min(x.output_value, axis=self.axis, keepdims=self.keepdims)
        return self.output_value

    def gradients_function(self):
        def grad_x(grad):
            x, = self.input_nodes
            if self.keepdims:
                return np.multiply(grad, np.equal(self.output_value, x.output_value))
            else:
                outshape = np.array(np.shape(x.output_value))
                outshape[self.axis] = 1
                t = np.shape(x.output_value)//outshape
                r = np.reshape(self.output_value, outshape)
                g = np.tile(r, t)
                grad = np.reshape(grad, outshape)
                eq = np.equal(g, x.output_value)
                return np.multiply(grad, eq)
        return [grad_x]

#pass
class Where(ArithmeticalOperation):
    def __init__(self, cond, x, y):
        super(self.__class__, self).__init__(cond, x, y)

    def compute_output(self):
        cond, x, y = self.input_nodes
        self.output_value = np.where(cond.output_value, x.output_value, y.output_value)
        return self.output_value

    def gradients_function(self):
        cond, x, y = self.input_nodes
        def grad_cond(grad):
            return np.zeros_like(cond.output_value)

        def grad_x(grad):
            return np.where(cond.output_value, grad, 0.0)

        def grad_y(grad):
            return np.where(cond.output_value, 0.0, grad)
        return [grad_cond, grad_x, grad_y]

#pass
class Transpose(ArithmeticalOperation):
    def __init__(self, x, axes=None):
        super(self.__class__, self).__init__(x)
        self.axes = axes
        self.cache = None

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = np.transpose(x.output_value, axes=self.axes)
        return self.output_value

    def gradients_function(self):
        def grad_x(grad):
            x, = self.input_nodes
            if self.cache is None:
                axes = self.axes
                if axes is None:
                    axes = reversed(range(np.ndim(x.output_value)))
                axes = list(axes)
                self.cache = np.argsort(axes)
            cache = self.cache
            return np.transpose(grad, cache)

        return [grad_x]

class ZeroPad(ArithmeticalOperation):
    def __init__(self, x, pad_width):
        super(self.__class__, self).__init__(x)
        self.pad_width = pad_width

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = np.pad(x.output_value, pad_width=self.pad_width, mode='constant')
        return self.output_value

    def gradients_function(self):
        def grad_x(grad):
            x, = self.input_nodes
            index = []
            #self.pad_width = [(l1,r1), (l2,r2), (l3,r3), ....]
            #x.output_value.shape = [d1, d2, d2,....]
            for p, r in zip(self.pad_width, np.shape(x.output_value)):
                s = slice(p[0], p[0] + r)
                index.append(s)
            return grad.__getitem__(tuple(index))

        return [grad_x]

class EdgePad(ArithmeticalOperation):
    #https://www.zhihu.com/people/yong-yuan-zai-ni-shen-hou-73/posts?page=2
    def __init__(self, x, pad_width):
        super(self.__class__, self).__init__(x)
        self.pad_width = pad_width

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = np.pad(x.output_value, pad_width=self.pad_width, mode='edge')
        return self.output_value

    def gradients_function(self):
            #self.pad_width = [(l1,r1), (l2,r2), (l3,r3), ....]
            #x.output_value.shape = [d1, d2, d2,....]
        def grad_x(grad):
            x, = self.input_nodes
            #grad = np.copy(grad)
            i = 0
            index = []
            temp_index = [slice(None, None, None) for i in range(len(np.shape(grad)))]
            for p, d in zip(self.pad_width, np.shape(x.output_value)):
                l,r = p
                if l != 0:
                    t = temp_index[i]
                    temp_index[i] = slice(0, l)
                    view = grad.__getitem__(tuple(temp_index))
                    s = np.sum(view, axis=i)
                    temp_index[i] = l
                    added = grad.__getitem__(tuple(temp_index))
                    np.add(s, added, added)
                    temp_index[i] = t
                if r != 0:
                    t = temp_index[i]
                    temp_index[i] = slice(l+p, -1)
                    view = grad.__getitem__(tuple(temp_index))
                    s = np.sum(view, axis=i)
                    temp_index[i] = l+p
                    added = grad.__getitem__(tuple(temp_index))
                    np.add(s, added, added)
                    temp_index[i] = t
                index.append(slice(l, l+p))
                i += 1
            return grad.__getitem__(tuple(index))

        return [grad_x]


class Normalization(ArithmeticalOperation):
    def __init__(self, cond, x, mean, std, momentum, eps):
        super(self.__class__, self).__init__(cond, x, mean, std, momentum, eps)
        mean.is_first = True
        std.is_first = True

        self.x_placeholder = Placeholder()#
        self.mean_placeholder = Placeholder()
        self.std_placeholder = Placeholder()
        self.momentum_placeholer = Placeholder()
        self.eps_placeholder = Placeholder()

        x_m = Mean(self.x_placeholder, axis=0)
        x_std0 = Subtract(self.x_placeholder, x_m)
        x_std1 = Square(x_std0)
        x_std2 = Mean(x_std1, axis=0)
        x_std3 = Add(x_std2, self.eps_placeholder)
        x_std4 = Sqrt(x_std3)
        true_x_hat = Divide(x_std0, x_std4)
        self.true_forward = Executor([true_x_hat, x_m, x_std4])
        self.true_backward = GradientForBranch(true_x_hat, [self.x_placeholder, self.eps_placeholder])

        false_x_hat = Divide(Subtract(self.x_placeholder, self.mean_placeholder), self.std_placeholder)
        self.false_forward = Executor([false_x_hat])
        self.false_backward = GradientForBranch(false_x_hat, [self.x_placeholder, self.mean_placeholder, self.std_placeholder])


    def compute_output(self):
        cond, x, mean, std, momentum, eps = self.input_nodes
        self.true_backward.clear_gradient_table()
        self.false_backward.clear_gradient_table()

        if cond.output_value:
            forward = self.true_forward
            feed_dict = {self.x_placeholder:x.output_value, self.eps_placeholder:eps.output_value}
            self.output_value, x_m, x_std = forward.run(feed_dict=feed_dict)
            if mean.is_first:
                mean.is_first = False
                mean.output_value = x_m
            else:
                np.add(np.multiply(momentum.output_value, mean.output_value), np.multiply(np.subtract(1.0, momentum.output_value), x_m), mean.output_value)
            if std.is_first:
                std.is_first = False
                std.output_value = x_std
            else:
                np.add(np.multiply(momentum.output_value, std.output_value), np.multiply(np.subtract(1.0, momentum.output_value), x_std), std.output_value)
        else:
            forward = self.false_forward
            feed_dict = {self.x_placeholder:x.output_value, self.mean_placeholder:mean.output_value, self.std_placeholder:std.output_value}
            self.output_value, = forward.run(feed_dict=feed_dict)
        return self.output_value

    def gradients_function(self):
        cond, x, mean, std, momentum, eps = self.input_nodes
        def grad_cond(grad):
            return np.zeros_like(cond.output_value)
        
        def grad_x(grad):
            if cond.output_value:
                return self.true_backward.get(0, grad)
            else:
                return self.false_backward.get(0, grad)

        def grad_mean(grad):
            if cond.output_value:
                return np.zeros_like(mean.output_value)
            else:
                return self.false_backward.get(1, grad)

        def grad_std(grad):
            if cond.output_value:
                return np.zeros_like(std.output_value)
            else:
                return self.false_backward.get(2, grad)

        def grad_momentum(grad):
            return np.zeros_like(momentum.output_value)

        def grad_eps(grad):
            if cond.output_value:
                return self.true_backward.get(1, grad)
            else:
                return np.zeros_like(eps.output_value)

        return [grad_cond, grad_x, grad_mean, grad_std, grad_momentum, grad_eps]

#pass
class Getitem(ArithmeticalOperation):
    def __init__(self, x, key):
        super(self.__class__, self).__init__(x)
        self.key = key

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = x.output_value.__getitem__(self.key)
        return self.output_value

    def gradients_function(self):
        def grad_x(grad):
            x, = self.input_nodes
            r = np.zeros_like(x.output_value)
            r.__setitem__(self.key, grad)          
            return r
        return [grad_x]

#pass
class Concatenate(ArithmeticalOperation):
    def __init__(self, lists, axis=0):
        super(self.__class__, self).__init__(*lists)
        self.axis = axis
        self.indexs = None

    def compute_output(self):
        #x, = self.input_nodes
        lists = [node.output_value for node in self.input_nodes]
        self.output_value = np.concatenate(lists, axis=self.axis)
        return self.output_value

    def gradient_function(self, i):  
        def grad_i(grad):
            if self.indexs is None:
                indexs = []
                axis = self.axis
                begin = 0
                end = begin
                for node in self.input_nodes:
                    shape = np.shape(node.output_value)
                    length = shape[axis]
                    index = [slice(None, None, None) for i in range(len(shape))]
                    end = begin + length
                    index[axis] = slice(begin, end, None)
                    indexs.append(tuple(index))
                    begin = end
                self.indexs = indexs
            return grad.__getitem__(self.indexs[i])
        return grad_i

    def gradients_function(self):
        funs = []
        for i in range(len(self.input_nodes)):
            funs.append(self.gradient_function(i))
        return funs

#pass
class ExpandDims(ArithmeticalOperation):
    def __init__(self, x, axis=0):
        super(self.__class__, self).__init__(x)
        self.axis = axis

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = np.expand_dims(x.output_value, axis=self.axis)
        return self.output_value

    def gradients_function(self):
        def grad_x(grad):
            x, = self.input_nodes
            return np.reshape(grad, np.shape(x.output_value))
        return [grad_x]

class Flip(ArithmeticalOperation):
    def __init__(self, x, axis=None):
        super(self.__class__, self).__init__(x)
        self.axis = axis

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = np.flip(x.output_value, axis=self.axis)
        return self.output_value

    def gradients_function(self):
        def grad_x(grad):
            x, = self.input_nodes
            return np.flip(grad, axis=self.axis)
        return [grad_x]

class Stack(ArithmeticalOperation):
    def __init__(self, x, axis=0):
        super(self.__class__, self).__init__(*x)
        self.axis = axis
        self.indexs = None

    def compute_output(self):
        arrays = [node.output_value for node in self.input_nodes]
        self.output_value = np.stack(arrays, axis=self.axis)
        return self.output_value

    def gradient_function(self, i):
        def grad_i(grad):
            if self.indexs is None:
                indexs = []
                axis = self.axis
                for j in range(len(self.input_nodes)):
                    index = [slice(None, None, None) for _ in range(len(np.shape(self.output_value)))]
                    index[axis] = j
                    indexs.append(tuple(index))
                self.indexs = indexs

            return grad.__getitem__(self.indexs[i])
        return grad_i
    
    def gradients_function(self):
        funs = []
        for i in range(len(self.input_nodes)):
            funs.append(self.gradient_function(i))
        return funs

class Swapaxes(ArithmeticalOperation):
    def __init__(self, x, axis1, axis2):
        super(self.__class__, self).__init__(x)
        self.axis1 = axis1
        self.axis2 = axis2

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = np.swapaxes(x.output_value, self.axis1, self.axis2)
        return self.output_value

    def gradients_function(self):
        def grad_x(grad):
            x, = self.input_nodes
            return np.swapaxes(x.output_value, self.axis1, self.axis2)
        return [grad_x]

class Moveaxis(ArithmeticalOperation):
    def __init__(self, x, source, destination):
        super(self.__class__, self).__init__(x)
        self.source = source
        self.destination = destination

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = np.moveaxis(x.output_value, self.source, self.destination)
        return self.output_value

    def gradients_function(self):
        def grad_x(grad):
            x, = self.input_nodes
            return np.moveaxis(grad, self.destination, self.source)
        return [grad_x]

class Rollaxis(ArithmeticalOperation):
    def __init__(self, x, axis, start=0):
        super(self.__class__, self).__init__(x)
        self.axis = axis
        self.start = start

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = np.rollaxis(x.output_value, self.axis, self.start)
        return self.output_value

    def gradients_function(self):
        def grad_x(grad):
            x, = self.input_nodes

            axis = self.axis
            L = len(np.shape(self.output_value))
            if axis < 0:
                axis += L
            start = self.start
            if start < 0:
                start += L
                
            if axis > start:
                return np.rollaxis(grad, start, axis+1)
            elif axis == start or axis + 1 == start:
                return grad
            else:
                return np.rollaxis(grad, start-1, axis)
        return [grad_x]

class Squeeze(ArithmeticalOperation):
    def __init__(self, x, axis=None):
        super(self.__class__, self).__init__(x)
        self.axis = axis

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = np.squeeze(x.output_value, axis=self.axis)
        return self.output_value

    def gradients_function(self):
        def grad_x(grad):
            x, = self.input_nodes
            return np.reshape(grad, np.shape(x.output_value))
        return [grad_x]

#实现可以再每一个维度上操作的conv 和 pooling, 类似于tensorflow



def im2col_1D(x, FW, stride):
    #x, kernel = self.input_nodes
    N, W, C = np.shape(x)
    #FW, _, output_channel = np.shape(kernel.output_value)
        
    #stride = self.stride

    out_w = (W - FW)//stride + 1
    col = np.empty((N, FW, out_w, C))
    img = x#x.output_value
    for x in range(FW):
        x_max = x + stride*out_w
        #N, FW, out_w, C  #N, W, C
        #col[:, x, :, :] = img[:, x:x_max:stride, :]
        index1 = (slice(None, None, None), x, slice(None, None, None), slice(None, None, None))
        index2 = (slice(None, None, None), slice(x, x_max, stride), slice(None, None, None))
        col.__setitem__(index1, img.__getitem__(index2))
        
    col = np.transpose(col, (0, 2, 1, 3))#N, out_w, FW, C
    #col = np.reshape(col, (N*out_w, -1))
    return col, out_w
def col2im_1D(x, col, out_w, FW, stride):
        #x, kernel = self.input_nodes
        N, W, C = np.shape(x)
        #_, out_w, output_channel = np.shape(self.output_value)
        #FW, _, _ = np.shape(kernel.output_value)

        #col = np.reshape(col, (N, out_w, FW, C))
        col = np.transpose(col, (0, 2, 1, 3))#N, FW, out_w, C
        #stride = self.stride

        img = np.zeros_like(x)#N, W, C
        for x in range(FW):
            x_max = x + stride*out_w
            #img[:, x:x_max:stride, :] += col[:, x, :, :]
            index1 = (slice(None, None, None), slice(x, x_max, stride), slice(None, None, None))
            index2 = (slice(None, None, None), x, slice(None, None, None), slice(None, None, None))
            np.add(img.__getitem__(index1), col.__getitem__(index2), img.__getitem__(index1))

        return img

#pass
class Conv1D(ArithmeticalOperation):
    #x : operation 及其子类, placeholder, constant, variable 
    #shape = (N, W, C)
    #kernel operation及其子类, placeholer, constant, variable
    #shape = (filter_W, C, output_channel)
    #returen shape = (N, (W - filter_w)/stride + 1, output_channel)
    def __init__(self, x, kernel, stride):
        super(self.__class__, self).__init__(x, kernel)
        self.stride = stride

    def im2col(self):
        x, kernel = self.input_nodes
        FW, C, output_channel = np.shape(kernel.output_value)
        col, out_w = im2col_1D(x.output_value, FW, self.stride)
        col = np.reshape(col, (-1, FW*C))
        return col, out_w, output_channel
    
    def compute_output(self):
        col, out_w, output_channel= self.im2col()
        self.col = col
        _, kernel = self.input_nodes
        col_w = np.reshape(kernel.output_value, (-1, output_channel))

        out = np.dot(col, col_w)
        self.output_value = np.reshape(out, (-1, out_w, output_channel))#np.reshape(out, (N, out_h, out_w, output_channel))
        return self.output_value

    def col2im(self, col):
        x, kernel = self.input_nodes
        N, W, C = np.shape(x.output_value)
        FW, _, _ = np.shape(kernel.output_value)
        _, out_w, _ = np.shape(self.output_value)
        col = np.reshape(col, (N, out_w, FW, C))
        return col2im_1D(x.output_value, col, out_w, FW, self.stride)

    def gradients_function(self):
        def grad_x(grad):
            x, kernel = self.input_nodes

            N, out_w, output_channel = np.shape(self.output_value)
            out = np.reshape(grad, (N*out_w, output_channel))
            col_w = np.reshape(kernel.output_value, (-1, output_channel))
            x_grad = np.dot(out, np.transpose(col_w))
            return self.col2im(x_grad)

        def grad_kernel(grad):
            x, kernel = self.input_nodes

            N, out_w, output_channel = np.shape(self.output_value)
            out = np.reshape(grad, (N*out_w, output_channel))
            col = self.col
            kernel_grad = np.dot(np.transpose(col), out)# -1, output_channel
            kernel_grad = np.reshape(kernel_grad, np.shape(kernel.output_value))
            return kernel_grad
    
        return [grad_x, grad_kernel]

#pass
class MaxPooling1D(ArithmeticalOperation):
    #x : operation 及其子类, placeholder, constant, variable 
    #shape = (N, W, C)
    #p_w
    #
    #return shape = (N, (W - p_w)//stride + 1, C)
    def __init__(self, x, p_w, stride):
        super(self.__class__, self).__init__(x)
        self.p_w = p_w
        self.stride = stride
        self.col = None

    def im2col(self):
        x, = self.input_nodes
        return im2col_1D(x.output_value, self.p_w, self.stride)#col, out_w
                                                               #N, out_w, FW, C
        
    def compute_output(self):
        col, out_w = self.im2col()
        x, = self.input_nodes
        N, W, C = np.shape(x.output_value)
        col = np.transpose(col, (0, 1, 3, 2))#N, out_w, C, FW
        col = np.reshape(col, (-1, self.p_w))
        self.col = col
        out = np.max(col, axis=1)
        self.output_value = np.reshape(out, (N, out_w, C))

    def col2im(self, col):
        x, = self.input_nodes
        N, out_w, C = np.shape(self.output_value)
        return col2im_1D(x.output_value, col, out_w, self.p_w, self.stride)

    def gradients_function(self):
        def grad_x(grad):
            x, = self.input_nodes

            N, out_w, C = np.shape(self.output_value)
            H = N*out_w*C
            dmax = np.zeros((H, self.p_w), dtype=x.output_value.dtype)
            arg_max = np.argmax(self.col, axis=1)
            dmax.__setitem__((range(H), np.ravel(arg_max)), np.ravel(grad))
            dmax = np.reshape(dmax, (N, out_w, C, self.p_w))#N, out_w, C, FW
            dmax = np.transpose(dmax, (0, 1, 3, 2))#N, out_w, FW, C
            return self.col2im(dmax)
        return [grad_x]

#pass
class MeanPooling1D(ArithmeticalOperation):
    #x : operation 及其子类, placeholder, constant, variable 
    #shape = (N, W, C)
    #p_w
    #
    #return shape = (N, (W - p_w)//stride + 1, C)
    def __init__(self, x, p_w, stride):
        super(self.__class__, self).__init__(x)
        self.p_w = p_w
        self.stride = stride
        self.col = None

    def im2col(self):
        x, = self.input_nodes
        return im2col_1D(x.output_value, self.p_w, self.stride)#col, out_w
                                                               #N, out_w, FW, C
        
    def compute_output(self):
        col, out_w = self.im2col()
        x, = self.input_nodes
        N, W, C = np.shape(x.output_value)
        col = np.transpose(col, (0, 1, 3, 2))#N, out_w, C, FW
        col = np.reshape(col, (-1, self.p_w))
        self.col = col
        out = np.mean(col, axis=1)
        self.output_value = np.reshape(out, (N, out_w, C))

    def col2im(self, col):
        x, = self.input_nodes
        N, out_w, C = np.shape(self.output_value)
        return col2im_1D(x.output_value, col, out_w, self.p_w, self.stride)

    def gradients_function(self):
        def grad_x(grad):
            x, = self.input_nodes

            N, out_w, C = np.shape(self.output_value)
            #grad.shape = (N, out_w, C)
            t_grad = np.divide(grad, self.p_w)
            t_grad = np.reshape(t_grad, (-1, 1))
            dmax = np.tile(t_grad, (1, self.p_w))
            dmax = np.reshape(dmax, (N, out_w, C, self.p_w))#N, out_w, C, FW
            dmax = np.transpose(dmax, (0, 1, 3, 2))#N, out_w, FW, C
            return self.col2im(dmax)
        return [grad_x]


def im2col_2D(x, FH, FW, stride_h, stride_w):
    N, H, W, C = np.shape(x)

    out_h = (H - FH)//stride_h + 1
    out_w = (W - FW)//stride_w + 1
    col = np.empty((N, FH, FW, out_h, out_w, C))
    img = x
    for y in range(FH):
        y_max = y + stride_h*out_h
        for x in range(FW):
            x_max = x + stride_w*out_w
                   #N, FH, FW, out_h, out_w, C     #N, H, W, C
                   #N, out_h, out_w, C             #N, out_h, out_w, C
                #col[:, y, x, :, :, :] = img[:, y:y_max:stride_h, x:x_max:stride_w, :]
            index1 = (slice(None, None, None), y, x, slice(None, None, None), slice(None, None, None),slice(None, None, None))
            index2 = (slice(None, None, None), slice(y, y_max, stride_h), slice(x, x_max, stride_w), slice(None, None, None))
            col.__setitem__(index1, img.__getitem__(index2) )

    col = np.transpose(col, (0, 3, 4, 1, 2, 5))#N, out_h, out_w, FH, FW, C
    return col, out_h, out_w

def col2im_2D(x, col, out_h, out_w, FH, FW, stride_h, stride_w):
    N, H, W, C = np.shape(x)

    col = np.transpose(col, (0, 3, 4, 1, 2, 5))

    img = np.zeros_like(x)
    for y in range(FH):
        y_max = y + stride_h*out_h
        for x in range(FW):
            x_max = x + stride_w*out_w
            index1 = (slice(None, None, None), slice(y, y_max, stride_h), slice(x, x_max, stride_w), slice(None, None, None))
            index2 = (slice(None, None, None), y, x, slice(None, None, None), slice(None, None, None), slice(None, None, None))
            np.add(img.__getitem__(index1), col.__getitem__(index2), img.__getitem__(index1))
            #img[:, y:y_max:stride_h, x:x_max:stride_w, :] += col[:, y, x, :, :, :]
    #N, FH, FW, out_h, out_w, C
    return img

#pass
class Conv2D(ArithmeticalOperation):
    #x : operation 及其子类, placeholder, constant, variable 
    #shape = (batch_size, H, W, C)
    #kernel operation及其子类, placeholer, constant, variable
    #shape = (FH, FW, C, output_channel)
    #
    def __init__(self, x, kernel, stride):
        super(self.__class__, self).__init__(x, kernel)
        self.stride_h = stride[0]
        self.stride_w = stride[1]
        self.col = None

    def im2col(self):
        x, kernel = self.input_nodes
        FH, FW, C, output_channel = np.shape(kernel.output_value)
        col, out_h, out_w = im2col_2D(x.output_value, FH, FW, self.stride_h, self.stride_w)
        col = np.reshape(col, (-1, FH*FW*C))
        return col, out_h, out_w, output_channel

    def compute_output(self):
        col, out_h, out_w, output_channel = self.im2col()
        self.col = col
        _, kernel = self.input_nodes
        col_w = np.reshape(kernel.output_value, (-1, output_channel))
        out = np.dot(col, col_w)
        self.output_value = np.reshape(out, (-1, out_h, out_w, output_channel))
        return self.output_value

    def col2im(self, col):
        x, kernel = self.input_nodes
        _, out_h, out_w, _ = np.shape(self.output_value)
        FH, FW, _, _ = np.shape(kernel.output_value)

        return col2im_2D(x.output_value, col, out_h, out_w, FH, FW, self.stride_h, self.stride_w)

    def gradients_function(self):
        def grad_x(grad):
            x, kernel = self.input_nodes

            N, out_h, out_w, output_channel = np.shape(self.output_value)
            FH, FW, C, _ = np.shape(kernel.output_value)
            out = np.reshape(grad, (N*out_h*out_w, output_channel))
            col_w = np.reshape(kernel.output_value, (-1, output_channel))
            x_grad = np.dot(out, np.transpose(col_w))
            x_grad = np.reshape(x_grad, (N, out_h, out_w, FH, FW, C))
            return self.col2im(x_grad)

        def grad_kernel(grad):
            x, kernel = self.input_nodes

            N, out_h, out_w, output_channel = np.shape(self.output_value)
            out = np.reshape(grad, (N*out_h*out_w, output_channel))
            col = self.col
            kernel_grad = np.dot(np.transpose(col), out)# -1, output_channel
            kernel_grad = np.reshape(kernel_grad, np.shape(kernel.output_value))
            return kernel_grad
    
        return [grad_x, grad_kernel]

#pass
class MaxPooling2D(ArithmeticalOperation):
    #x : operation 及其子类, placeholder, constant, variable 
    #shape = (N, H, W, C)
    #pooling_shape = (p_h, p_w)
    #stride = (stride_h, stride_w)
    #return shape = (N, (H - p_h)//stride_h + 1, (W - p_w)//stride_w + 1, C)
    def __init__(self, x, pooling_shape, stride):
        super(self.__class__, self).__init__(x)
        self.stride_h = stride[0]
        self.stride_w = stride[1]
        self.p_h = pooling_shape[0]
        self.p_w = pooling_shape[1]
        self.col = None

    def im2col(self):
        x, = self.input_nodes
        return im2col_2D(x.output_value, self.p_h, self.p_w, self.stride_h, self.stride_w)
        

    def compute_output(self):
        col, out_h, out_w = self.im2col()
        x, = self.input_nodes
        N, _, _, C = np.shape(x.output_value)#C = np.shape(x.output_value)[3]
        #N, out_h, out_w, FH, FW, C
        t = np.transpose(col, (0, 1, 2, 5, 3, 4))#N, out_h, out_w, C, FH, FW
        t = np.reshape(t, (-1, self.p_h*self.p_w))
        self.col = t#(N*out_h*out_w*C, FH*FW)
        out = np.max(t, axis=1)
        self.output_value = np.reshape(out, (N, out_h, out_w, C))#(N, out_h, out_w, C)
        return self.output_value

    def col2im(self, col):
        x, = self.input_nodes
        N, out_h, out_w, C = np.shape(self.output_value)
        return col2im_2D(x.output_value, col, out_h, out_w, self.p_h, self.p_w, self.stride_h, self.stride_w)

    def gradients_function(self):
        def grad_x(grad):
            x, = self.input_nodes

            N, out_h, out_w, C = np.shape(self.output_value)
            p_size = self.p_h*self.p_w
            H = N*out_h*out_w*C
            dmax = np.zeros((H, p_size), dtype=x.output_value.dtype)
            arg_max = np.argmax(self.col, axis=1)
            dmax.__setitem__((range(H), np.ravel(arg_max)), np.ravel(grad))#(N*out_h*out_w*C, FH*FW)
            #dmax[range(arg_max.size), arg_max.ravel()] = grad.ravel()
            dmax = np.reshape(dmax, (N, out_h, out_w, C, self.p_h, self.p_w))#(N, out_h, out_w, C, FH, FW)
            dmax = np.transpose(dmax, (0, 1, 2, 4, 5, 3))#(N, out_h, out_w, FH, FW, C)
            return self.col2im(dmax)

        return [grad_x]

#pass
class MeanPooling2D(ArithmeticalOperation):
    #x : operation 及其子类, placeholder, constant, variable 
    #shape = (N, H, W, C)
    #pooling_shape = (p_h, p_w)
    #stride = (stride_h, stride_w)
    #return shape = (N, (H - p_h)//stride_h + 1, (W - p_w)//stride_w + 1, C)
    def __init__(self, x, pooling_shape, stride):
        super(self.__class__, self).__init__(x)
        self.stride_h = stride[0]
        self.stride_w = stride[1]
        self.p_h = pooling_shape[0]
        self.p_w = pooling_shape[1]
        self.col = None

    def im2col(self):
        x, = self.input_nodes
        return im2col_2D(x.output_value, self.p_h, self.p_w, self.stride_h, self.stride_w)
        

    def compute_output(self):
        col, out_h, out_w = self.im2col()
        x, = self.input_nodes
        N, _, _, C = np.shape(x.output_value)#C = np.shape(x.output_value)[3]
        #N, out_h, out_w, FH, FW, C
        t = np.transpose(col, (0, 1, 2, 5, 3, 4))#N, out_h, out_w, C, FH, FW
        t = np.reshape(t, (-1, self.p_h*self.p_w))
        self.col = t#(N*out_h*out_w*C, FH*FW)
        out = np.mean(t, axis=1)
        self.output_value = np.reshape(out, (N, out_h, out_w, C))#(N, out_h, out_w, C)
        return self.output_value

    def col2im(self, col):
        x, = self.input_nodes
        N, out_h, out_w, C = np.shape(self.output_value)
        return col2im_2D(x.output_value, col, out_h, out_w, self.p_h, self.p_w, self.stride_h, self.stride_w)

    def gradients_function(self):
        def grad_x(grad):
            x, = self.input_nodes

            N, out_h, out_w, C = np.shape(self.output_value)
            dmax = np.divide(grad, self.p_h*self.p_w)
            dmax = np.reshape(dmax, (-1, 1))
            dmax = np.tile(dmax, (1, self.p_h*self.p_w))
            dmax = np.reshape(dmax, (N, out_h, out_w, C, self.p_h, self.p_w))#(N, out_h, out_w, C, FH, FW)
            dmax = np.transpose(dmax, (0, 1, 2, 4, 5, 3))#(N, out_h, out_w, FH, FW, C)
            return self.col2im(dmax)

        return [grad_x]

def im2col_3D(x, FD, FH, FW, stride_d, stride_h, stride_w):
    N, D, H, W, C = np.shape(x)
        
    out_d = (D - FD)//stride_d + 1
    out_h = (H - FH)//stride_h + 1
    out_w = (W - FW)//stride_w + 1

    col = np.empty((N, FD, FH, FW, out_d, out_h, out_w, C))
    img = x
    for z in range(FD):
        z_max = z + stride_d*out_d
        for y in range(FH):
            y_max = y + stride_h*out_h
            for x in range(FW):
                x_max = x + stride_w*out_w
                #col[:, z, y, x, :, :, :, :] = img[:, z:z_max:stride_d, y:y_max:stride_h, x:x_max:stride_w, :]
                index1 = (slice(None, None, None), z, y, x, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, None))
                index2 = (slice(None, None, None), slice(z, z_max, stride_d), slice(y, y_max, stride_h), slice(x, x_max, stride_w), slice(None, None, None))
                col.__setitem__(index1, img.__getitem__(index2))
    #N, FD, FH, FW, out_d, out_h, out_w, C
    col = np.transpose(col, (0, 4, 5, 6, 1, 2, 3, 7))#N, out_d, out_h, out_w, FD, FH, FW, C
    return col, out_d, out_h, out_w
def col2im_3D(x, col, out_d, out_h, out_w, FD, FH, FW, stride_d, stride_h, stride_w):
    N, D, H, W, C = np.shape(x)
    col = np.transpose(col, (0, 4, 5, 6, 1, 2, 3, 7))#N, FD, FH, FW, out_d, out_h, out_w, C

    img = np.zeros_like(x)#N, D, H, W, C
    for z in range(FD):
        z_max = z + stride_d*out_d
        for y in range(FH):
            y_max = y + stride_h*out_h
            for x in range(FW):
                x_max = x + stride_w*out_w
                #img[:, z:z_max:stride_d, y:y_max:stride_h, x:x_max:stride_w, :] += col[:, z, y, x, :, :, :, :]
                index1 = (slice(None, None, None), slice(z, z_max, stride_d), slice(y, y_max, stride_h), slice(x, x_max, stride_w), slice(None, None, None))
                index2 = (slice(None, None, None), z, y, x, slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, None, None))
                np.add(img.__getitem__(index1), col.__getitem__(index2), img.__getitem__(index1))

    return img

#semi pass
class Conv3D(ArithmeticalOperation):
    #x : operation 及其子类, placeholder, constant, variable 
    #shape = (N, D, H, W, C)
    #kernel operation及其子类, placeholer, constant, variable
    #shape = (FD, FH, FW, C, output_channel)
    #return shape = (N, out_d, out_w, out_h, output_channel)
    def __init__(self, x, kernel, stride):
        super(self.__class__, self).__init__(x, kernel)
        self.stride_d = stride[0]
        self.stride_h = stride[1]
        self.stride_w = stride[2]
        self.col = None

    def im2col(self):
        x, kernel = self.input_nodes
        FD, FH, FW, C, output_channel = np.shape(kernel.output_value)
        col, out_d, out_h, out_w = im2col_3D(x.output_value, FD, FH, FW, self.stride_d, self.stride_h, self.stride_w)
        col = np.reshape(col, (-1, FD*FH*FW*C))
        return col, out_d, out_h, out_w, output_channel
    

    def compute_output(self):
        col, out_d, out_h, out_w, output_channel = self.im2col()
        self.col = col
        _, kernel = self.input_nodes
        col_w = np.reshape(kernel.output_value, (-1, output_channel))
        out = np.dot(col, col_w)
        self.output_value = np.reshape(out, (-1, out_d, out_h, out_w, output_channel))
        return self.output_value

    def col2im(self, col):
        x, kernel = self.input_nodes
        N, out_d, out_h, out_w, output_channel = np.shape(self.output_value)
        FD, FH, FW, C, _ = np.shape(kernel.output_value)
        col = np.reshape(col, (N, out_d, out_h, out_w, FD, FH, FW, C))
        return col2im_3D(x.output_value, col, out_d, out_h, out_w, FD, FH, FW, self.stride_d, self.stride_h, self.stride_w)

    def gradients_function(self):
        def grad_x(grad):
            x, kernel = self.input_nodes

            N, out_d, out_h, out_w, output_channel = np.shape(self.output_value)
            out = np.reshape(grad, (N*out_d*out_h*out_w, output_channel))
            col_w = np.reshape(kernel.output_value, (-1, output_channel))
            x_grad = np.dot(out, np.transpose(col_w))
            return self.col2im(x_grad)

        def grad_kernel(grad):
            x, kernel = self.input_nodes

            N, out_d, out_h, out_w, output_channel = np.shape(self.output_value)
            out = np.reshape(grad, (N*out_d*out_h*out_w, output_channel))
            col = self.col
            kernel_grad = np.dot(np.transpose(col), out)# -1, output_channel
            kernel_grad = np.reshape(kernel_grad, np.shape(kernel.output_value))
            return kernel_grad
    
        return [grad_x, grad_kernel]

#semi pass
class MaxPooling3D(ArithmeticalOperation):
    #x : operation 及其子类, placeholder, constant, variable 
    #shape = (N, D, H, W, C)
    #pooling_shape = (p_d, p_h, p_w)
    #stride = (stride_d, stride_h, stride_w)
    #return shape = (N, (D - p_d)//stride_d + 1, (H - p_h)//stride_h + 1, (W - p_w)//stride_w + 1, C)
    def __init__(self, x, pooling_shape, stride):
        super(self.__class__, self).__init__(x)
        self.stride_d = stride[0]
        self.stride_h = stride[1]
        self.stride_w = stride[2]
        self.p_d = pooling_shape[0]
        self.p_h = pooling_shape[1]
        self.p_w = pooling_shape[2]
        self.col = None

    def im2col(self):
        x, = self.input_nodes
        return im2col_3D(x.output_value, self.p_d, self.p_h, self.p_w, self.stride_d, self.stride_h, self.stride_w)

    def compute_output(self):
        col, out_d, out_h, out_w = self.im2col()#N, out_d, out_h, out_w, FD, FH, FW, C
        x, = self.input_nodes
        N, D, H, W, C = np.shape(x.output_value)
        col = np.transpose(col, (0, 1, 2, 3, 7, 4, 5, 6))#N, out_d, out_h, out_w, C, FD, FH, FW
        col = np.reshape(col, (-1, self.p_d*self.p_h*self.p_w))
        self.col = col
        out = np.max(col, axis=1)
        out = np.reshape(out, (N, out_d, out_h, out_w, C))
        self.output_value = out
        return self.output_value

    def col2im(self, col):
        x, = self.input_nodes
        N, out_d, out_h, out_w, C = np.shape(self.output_value)
        return col2im_3D(x.output_value, col, out_d, out_h, out_w, self.p_d, self.p_h, self.p_w, self.stride_d, self.stride_h, self.stride_w)

    def gradients_function(self):
        def grad_x(grad):
            x, = self.input_nodes

            N, out_d, out_h, out_w, C = np.shape(self.output_value)
            p_size = self.p_d*self.p_h*self.p_w
            H = N*out_d*out_h*out_w*C
            dmax = np.zeros((H, p_size), dtype=x.output_value.dtype)
            arg_max = np.argmax(self.col, axis=1)
            dmax.__setitem__((range(H), np.ravel(arg_max)), np.ravel(grad))#(N*out_d*out_h*out_w*C, p_d*p_h*p_w)
            #dmax[range(arg_max.size), arg_max.ravel()] = grad.ravel()
            dmax = np.reshape(dmax, (N, out_d, out_h, out_w, C, self.p_d, self.p_h, self.p_w))#(N, out_d, out_h, out_w, C, p_d, p_h, p_w)
            dmax = np.transpose(dmax, (0, 1, 2, 3, 5, 6, 7, 4))#(N, out_d, out_h, out_w, p_d, p_h, p_w, C)
            return self.col2im(dmax)

        return [grad_x]

#tensorflow has some errors so cannot test this class
class MeanPooling3D(ArithmeticalOperation):
    #x : operation 及其子类, placeholder, constant, variable 
    #shape = (N, D, H, W, C)
    #pooling_shape = (p_d, p_h, p_w)
    #stride = (stride_d, stride_h, stride_w)
    #return shape = (N, (D - p_d)//stride_d + 1, (H - p_h)//stride_h + 1, (W - p_w)//stride_w + 1, C)
    def __init__(self, x, pooling_shape, stride):
        super(self.__class__, self).__init__(x)
        self.stride_d = stride[0]
        self.stride_h = stride[1]
        self.stride_w = stride[2]
        self.p_d = pooling_shape[0]
        self.p_h = pooling_shape[1]
        self.p_w = pooling_shape[2]
        self.col = None

    def im2col(self):
        x, = self.input_nodes
        return im2col_3D(x.output_value, self.p_d, self.p_h, self.p_w, self.stride_d, self.stride_h, self.stride_w)

    def compute_output(self):
        col, out_d, out_h, out_w = self.im2col()#N, out_d, out_h, out_w, FD, FH, FW, C
        x, = self.input_nodes
        N, D, H, W, C = np.shape(x.output_value)
        col = np.transpose(col, (0, 1, 2, 3, 7, 4, 5, 6))#N, out_d, out_h, out_w, C, FD, FH, FW
        col = np.reshape(col, (-1, self.p_d*self.p_h*self.p_w))
        self.col = col
        out = np.mean(col, axis=1)
        out = np.reshape(out, (N, out_d, out_h, out_w, C))
        self.output_value = out
        return self.output_value

    def col2im(self, col):
        x, = self.input_nodes
        N, out_d, out_h, out_w, C = np.shape(self.output_value)
        return col2im_3D(x.output_value, col, out_d, out_h, out_w, self.p_d, self.p_h, self.p_w, self.stride_d, self.stride_h, self.stride_w)

    def gradients_function(self):
        def grad_x(grad):
            x, = self.input_nodes

            N, out_d, out_h, out_w, C = np.shape(self.output_value)
            p_size = self.p_d*self.p_h*self.p_w
            dmax = np.divide(grad, p_size)
            dmax = np.reshape(dmax, (-1, 1))
            dmax = np.tile(dmax, (1, p_size))
            dmax = np.reshape(dmax, (N, out_d, out_h, out_w, C, self.p_d, self.p_h, self.p_w))#(N, out_d, out_h, out_w, C, p_d, p_h, p_w)
            dmax = np.transpose(dmax, (0, 1, 2, 3, 5, 6, 7, 4))#(N, out_d, out_h, out_w, p_d, p_h, p_w, C)
            return self.col2im(dmax)
        return [grad_x]


class GradientForBranch(object):
    #objective 是目标函数
    #variables 是需要求导的变量的列表
    def __init__(self, objective, variables):
        self.objective = objective
        self.variables = variables
        self.grad_funs = {}
        self.topos = []
        self.can_be_computed = None
        self.init()
        self.grad_table = {}
        
    def init(self):
        target_op = self.objective
        queue = Queue()
        visited = set()
        queue.put(target_op)        
        visited.add(target_op)
        while not queue.empty():
            node = queue.get()

            if isinstance(node, ArithmeticalOperation):
                for input_node in node.input_nodes:
                    if input_node not in visited:
                        visited.add(input_node)
                        queue.put(input_node)
        self.can_be_computed = visited

        for var in self.variables:
            if var not in visited:
                self.topos.append(None)
            else:
                topo = list(reversed(topo_sort(visited, var)))
                self.topos.append(topo)
                
                for node in topo:
                    if isinstance(node, ArithmeticalOperation):
                        if self.grad_funs.get(node) is None:
                            self.grad_funs[node] = node.gradients_function()

    def clear_gradient_table(self):
        self.grad_table.clear()

    def get(self, index, grad=None):
        target_op = self.objective
        grad_table = self.grad_table
        if grad is None:
            grad_table[target_op] = np.ones_like(target_op.output_value)
        else:
            grad_table[target_op] = grad

        variable = self.variables[index]
        topo = self.topos[index]
        grad = grad_table.get(variable)
        if grad is None:
            if topo is None:
                grad = np.zeros_like(variable.output_value)
                grad_table[variable] = grad
            else:
                for v in topo:
                    grad_v = grad_table.get(v)
                    if grad_v is None:
                        for output_node in v.output_nodes:
                            if not isinstance(output_node, ArithmeticalOperation):
                                continue
                            if output_node in self.can_be_computed:
                                grad_output = grad_table.get(output_node)
                                funs = self.grad_funs.get(output_node)
                                grad_v_partial = None
                                if len(funs) == 1:
                                    grad_v_partial = funs[0](grad_output)
                                else:
                                    index = output_node.input_nodes.index(v)
                                    grad_v_partial = funs[index](grad_output)
                                if grad_v is None:
                                    grad_v = grad_v_partial
                                else:
                                    grad_v = np.add(grad_v, grad_v_partial)
                        grad_table[v] = grad_v
                    grad = grad_v
        return grad

#pass
class Branch(ArithmeticalOperation):
    def __init__(self, cond, f0, f1, *nodes):
        super(self.__class__, self).__init__(cond, *nodes)
        true_placeholders = [Placeholder() for i in range(len(nodes))]
        self.true_placeholders = true_placeholders
        self.true_output = f0(*true_placeholders)
        self.true_executor = Executor([self.true_output])
        self.true_gradient = GradientForBranch(self.true_output, self.true_placeholders)


        false_placeholders = [Placeholder() for i in range(len(nodes))]
        self.false_placeholders = false_placeholders
        self.false_output = f1(*false_placeholders)
        self.false_executor = Executor([self.false_output])
        self.false_gradient = GradientForBranch(self.false_output, self.false_placeholders)


    def compute_output(self):
        cond = self.input_nodes[0]
        nodes = self.input_nodes[1:]
        if cond.output_value:
            feed_dict = {}
            for placeholer, node in zip(self.true_placeholders, nodes):
                feed_dict[placeholer] = node.output_value
            self.output_value, = self.true_executor.run(feed_dict)
        else:
            feed_dict = {}
            for placeholer, node in zip(self.false_placeholders, nodes):
                feed_dict[placeholer] = node.output_value
            self.output_value, = self.false_executor.run(feed_dict)
        self.true_gradient.clear_gradient_table()
        self.false_gradient.clear_gradient_table()
        return self.output_value

    def gradient_function(self, index, cond):
        def grad_i(grad):
            if cond.output_value:
                return self.true_gradient.get(index, grad)
            else:
                return self.false_gradient.get(index, grad)
        return grad_i

    def gradients_function(self):
        cond = self.input_nodes[0]
        def grad_cond(grad):
            return np.zeros_like(cond.output_value)
        grads = [grad_cond]
        for i in range(1, len(self.input_nodes)):
            grads.append(self.gradient_function(i - 1, cond))        
        return grads        

""""
LogicalOperation
"""
#pass
class Argmax(LogicalOperation):
    def __init__(self, x, axis=None):
        super(self.__class__, self).__init__(x)
        self.axis = axis
    
    def compute_output(self):
        x, = self.input_nodes
        self.output_value = np.argmax(x.output_value, axis=self.axis)
        return self.output_value

#pass
class Argmin(LogicalOperation):
    def __init__(self, x, axis=None):
        super(self.__class__, self).__init__(x)
        self.axis = axis

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = np.argmin(x.output_value, axis=self.axis)
        return self.output_value

#pass
class Greater(LogicalOperation):
    def __init__(self, x, y):
        super(self.__class__, self).__init__(x, y)

    def compute_output(self):
        x, y = self.input_nodes
        self.output_value = np.greater(x.output_value, y.output_value)
        return self.output_value

#pass
class GreaterEqual(LogicalOperation):
    def __init__(self, x, y):
        super(self.__class__, self).__init__(x, y)

    def compute_output(self):
        x, y = self.input_nodes
        self.output_value = np.greater_equal(x.output_value, y.output_value)
        return self.output_value

#pass
class Less(LogicalOperation):
    def __init__(self, x, y):
        super(self.__class__, self).__init__(x, y)

    def compute_output(self):
        x, y = self.input_nodes
        self.output_value = np.less(x.output_value, y.output_value)
        return self.output_value

#pass
class LessEqual(LogicalOperation):
    def __init__(self, x, y):
        super(self.__class__, self).__init__(x, y)

    def compute_output(self):
        x, y = self.input_nodes
        self.output_value = np.less_equal(x.output_value, y.output_value)
        return self.output_value

#pass
class Equal(LogicalOperation):
    def __init__(self, x, y):
        super(self.__class__, self).__init__(x, y)

    def compute_output(self):
        x, y = self.input_nodes
        self.output_value = np.equal(x.output_value, y.output_value)
        return self.output_value

#pass
class NotEqual(LogicalOperation):
    def __init__(self, x, y):
        super(self.__class__, self).__init__(x, y)

    def compute_output(self):
        x, y = self.input_nodes
        self.output_value = np.not_equal(x.output_value, y.output_value)
        return self.output_value   

class Any(LogicalOperation):
    def __init__(self, x):
        super(self.__class__, self).__init__(x)

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = np.any(x.output_value)
        return self.output_value  

class All(LogicalOperation):
    def __init__(self, x):
        super(self.__class__, self).__init__(x)

    def compute_output(self):
        x, = self.input_nodes
        self.output_value = np.all(x.output_value)
        return self.output_value  

from queue import Queue

def input_degree(n, visited, nodes):
    if isinstance(n, Operation):
        for n0 in n.input_nodes:
            if n0 in nodes:
                if n0 not in visited:
                    return 1
    return 0
def topo_sort(nodes, dx):
    queue = Queue()
    visited = set()
    queue.put(dx)
    visited.add(dx)
    while not queue.empty():
        current = queue.get()
        for node in current.output_nodes:
            if isinstance(node, ArithmeticalOperation):
                if node in nodes and node not in visited:
                    queue.put(node)
                    visited.add(node)
    
    L = []
    current_visited = set()
    queue.put(dx)
    current_visited.add(dx)
    while not queue.empty():
        node = queue.get()
        L.append(node)
        for output_node in node.output_nodes:
            if isinstance(output_node, ArithmeticalOperation):
                if output_node in visited and output_node not in current_visited:
                    degree = input_degree(output_node, current_visited, visited)
                    if degree == 0:
                        queue.put(output_node)
                        current_visited.add(output_node)
    return L




class Gradient(object):
    #objective 是目标函数
    #variables 是需要求导的变量的列表
    def __init__(self, objective, variables):
        self.objective = objective
        self.variables = variables
        self.grad_funs = {}
        self.topos = []
        self.can_be_computed = None
        self.init()
        
    def init(self):
        target_op = self.objective
        queue = Queue()
        visited = set()
        queue.put(target_op)        
        visited.add(target_op)
        while not queue.empty():
            node = queue.get()

            if isinstance(node, ArithmeticalOperation):
                for input_node in node.input_nodes:
                    if input_node not in visited:
                        visited.add(input_node)
                        queue.put(input_node)
        self.can_be_computed = visited


        for var in self.variables:
            if var not in visited:
                self.topos.append(None)
            else:
                topo = list(reversed(topo_sort(visited, var)))
                self.topos.append(topo)
                
                for node in topo:
                    if isinstance(node, ArithmeticalOperation):
                        if self.grad_funs.get(node) is None:
                            self.grad_funs[node] = node.gradients_function()
               
    def get(self, grad=None):
        target_op = self.objective
        grad_table = {}
        if grad is None:
            grad_table[target_op] = np.ones_like(target_op.output_value)
        else:
            grad[target_op] = grad

        results = []
        for i, topo in enumerate(self.topos):
            variable = self.variables[i]
            grad = grad_table.get(variable)
            if grad is None:
                if topo is None:
                    grad = np.zeros_like(variable.output_value)
                    grad_table[variable] = grad
                else:
                    for v in topo:
                        grad_v = grad_table.get(v)
                        if grad_v is None:
                            for output_node in v.output_nodes:
                                if not isinstance(output_node, ArithmeticalOperation):
                                    continue
                                if output_node in self.can_be_computed:
                                    grad_output = grad_table.get(output_node)
                                    funs = self.grad_funs.get(output_node)
                                    grad_v_partial = None
                                    if len(funs) == 1:
                                        grad_v_partial = funs[0](grad_output)
                                    else:
                                        index = output_node.input_nodes.index(v)
                                        grad_v_partial = funs[index](grad_output)
                                    if grad_v is None:
                                        grad_v = grad_v_partial
                                    else:
                                        grad_v = np.add(grad_v, grad_v_partial)
                            grad_table[v] = grad_v
                        grad = grad_v
                    
            results.append(grad)
        return results


            
def output_degree(n, nodes, visited):
    for node in n.output_nodes:
        if node in nodes and node not in visited:
            return 1
    return 0

def _get_prerequisite(operation):

    nodes = set()
    queue = Queue()
    queue.put(operation)
    nodes.add(operation)
    while not queue.empty():
        current = queue.get()

        if isinstance(current, Operation):
            for node in current.input_nodes:
                if node not in nodes:
                    queue.put(node)
                    nodes.add(node)
    
    L = []
    visited = set()
    L.append(operation)
    visited.add(operation)
    queue.put(operation)
    while not queue.empty():
        current = queue.get() 
        if isinstance(current, Operation):
            for node in current.input_nodes:
                if node not in visited:
                    if output_degree(node, nodes, visited) == 0:
                        L.append(node)
                        queue.put(node)
                        visited.add(node)
    return list(reversed(L))
class Executor(object):
    def __init__(self, objectives):
        self.objectives = objectives
        self.postorder_nodes = []
        self.init()
        
    def init(self):
        for operation in self.objectives:
            postorder_nodes = _get_prerequisite(operation)
            self.postorder_nodes.append(postorder_nodes)
        
    def run(self, feed_dict=None):
        computed_set = set()
        results = []
        for postorder_nodes in self.postorder_nodes:
            for node in postorder_nodes:
                if node not in computed_set:                    
                    if type(node) is Placeholder:
                        node.output_value = feed_dict[node]
                    else:
                        node.compute_output()
                    computed_set.add(node)
            results.append(postorder_nodes[-1].output_value)
        return results
