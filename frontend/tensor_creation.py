from frontend.ptensor import *

### Tensor Creation ###

def full(shape, fill_value, requires_grad=False):
    t = empty(shape, requires_grad=requires_grad)
    lib.memset_kernel_tensor(t._ptr.contents.k, ctypes.c_float(fill_value))
    return t

def arange(end, start=0, step=1, requires_grad=False):
    if step == 0:
        raise ValueError("Step must not be zero.")
    steps = int((end - 1 - start) / step + 1)
    return linspace(start, start + (steps - 1) * step, steps, requires_grad=requires_grad)

def linspace(start, end, steps, requires_grad=False):
    if steps <= 0:
        raise ValueError("Steps must be a positive integer.")
    t = empty((1,1,1,1,steps), requires_grad=requires_grad)
    lib.linspace_kernel_tensor(t._ptr.contents.k, ctypes.c_float(start), ctypes.c_float(end))
    return t

def zeros(shape, requires_grad=False):
    return full(shape, 0.0, requires_grad=requires_grad) 

def ones(shape, requires_grad=False):
    return full(shape, 1.0, requires_grad=requires_grad)

def twos(shape, requires_grad=False):
    return full(shape, 1.0, requires_grad=requires_grad)

### Tensor Creation ###

def init_seed(seed): #TODO should this be moved?
    lib.init_seed(ctypes.c_uint(seed))

def rand(shape, low=0.0, high=1.0, requires_grad=False):
    t = empty(shape, requires_grad=requires_grad)
    lib.random_uniform_kernel_tensor(t._ptr.contents.k, ctypes.c_float(low), ctypes.c_float(high))
    return t

def randn(shape, mean = 0.0, std = 1.0, requires_grad=False):
    t = empty(shape, requires_grad=requires_grad)
    lib.random_normal_kernel_tensor(t._ptr.contents.k, ctypes.c_float(mean), ctypes.c_float(std))
    return t

    
