import ctypes
import os
import platform


SYSTEM = platform.system()
if SYSTEM == "Darwin":
    LIB_FILE = "liblightlemur.dylib"  
elif SYSTEM == "Windows":
    LIB_FILE = "liblightlemur.dll"   
else:
    LIB_FILE = "liblightlemur.so"    

lib_path = os.path.join(
    os.path.dirname(__file__),  
    "..",                      
    LIB_FILE
)

lib = ctypes.CDLL(lib_path)

class lemur_float(ctypes.c_float): #TODO
    pass

class KernelTensor(ctypes.Structure):
    _fields_ = [
        ("array",    ctypes.POINTER(lemur_float)),  
        ("length",   ctypes.c_size_t),
        ("shape",    ctypes.c_size_t * 5),            
        ("stride",   ctypes.c_int64  * 5),            
        ("computed", ctypes.c_bool),
        ("shallow",       ctypes.c_bool),  
    ]

class Tensor(ctypes.Structure):
    pass 

class Expression(ctypes.Structure):
    _fields_ = [
        ("t0",    ctypes.POINTER(Tensor)),  
        ("t1",    ctypes.POINTER(Tensor)),
        ("backward_func", ctypes.c_int),            
    ] 

class Tensor(ctypes.Structure):
    _fields_ = [
        ("k",             ctypes.POINTER(KernelTensor)), 
        ("comes_from",    ctypes.POINTER(Expression)),    
        ("requires_grad", ctypes.c_bool),
        ("grad",          ctypes.POINTER(KernelTensor)),  
    ]

class Parameter(ctypes.Structure):
    _fields_ = [
        ("tensor_ptr",    ctypes.POINTER(Tensor)), 
        ("first_moment",  ctypes.c_float), 
        ("second_moment", ctypes.c_float),
    ]

#from parameter.h
lib.create_parameter.argtypes = [ctypes.POINTER(Tensor)]
lib.create_parameter.restype = ctypes.POINTER(Parameter)

lib.free_parameter.argtypes = [ctypes.POINTER(Parameter)]
lib.free_parameter.restype = None

#from interface.h
lib.compile.argtypes = [ctypes.POINTER(Tensor)] 
lib.compile.restype = None


lib.tensor_from.argtypes = [ctypes.POINTER(KernelTensor), ctypes.POINTER(Expression), ctypes.c_bool, ctypes.POINTER(KernelTensor)] 
lib.tensor_from.restype = ctypes.POINTER(Tensor)

lib.contiguous_deepcopy_kernel_tensor.argtypes = [ctypes.POINTER(KernelTensor)] 
lib.contiguous_deepcopy_kernel_tensor.restype = ctypes.POINTER(KernelTensor)

lib.is_contiguous.argtypes = [ctypes.POINTER(KernelTensor)] 
lib.is_contiguous.restype = ctypes.c_bool 

lib.random_uniform_kernel_tensor.argtypes = [ctypes.POINTER(KernelTensor), ctypes.c_float, ctypes.c_float] #lemur_float
lib.random_uniform_kernel_tensor.restype = None 

lib.random_normal_kernel_tensor.argtypes = [ctypes.POINTER(KernelTensor), ctypes.c_float, ctypes.c_float] #lemur_float
lib.random_normal_kernel_tensor.restype = None

lib.linspace_kernel_tensor.argtypes = [ctypes.POINTER(KernelTensor), ctypes.c_float, ctypes.c_float] #lemur_float
lib.linspace_kernel_tensor.restype = None

#void init_seed(unsigned int seed);
lib.init_seed.argtypes = [ctypes.c_uint]
lib.init_seed.restype = None

lib.memset_kernel_tensor.argtypes = [ctypes.POINTER(KernelTensor), ctypes.c_float] #lemur_float
lib.memset_kernel_tensor.restype = None 

lib.get_op_name.argtypes = [ctypes.c_int]
lib.get_op_name.restype  = ctypes.c_char_p

# tensor* empty_tensor(size_t shape[5], bool requires_grad, bool retains_grad);
lib.empty_tensor.argtypes = [(ctypes.c_size_t * 5), ctypes.c_bool, ctypes.c_bool]
lib.empty_tensor.restype  = ctypes.POINTER(Tensor)

# void free_tensor(tensor* t);
lib.free_tensor.argtypes = [ctypes.POINTER(Tensor)]
lib.free_tensor.restype  = None

# void backward(tensor* t);
lib.backward.argtypes = [ctypes.POINTER(Tensor)]
lib.backward.restype  = None

# tensor* sub(tensor* t0, tensor* t1, bool retain_grad);
lib.sub.argtypes = [ctypes.POINTER(Tensor), ctypes.POINTER(Tensor), ctypes.c_bool]
lib.sub.restype  = ctypes.POINTER(Tensor)

# tensor* mul(tensor* t0, tensor* t1, bool retain_grad);
lib.mul.argtypes = [ctypes.POINTER(Tensor), ctypes.POINTER(Tensor), ctypes.c_bool]
lib.mul.restype  = ctypes.POINTER(Tensor)

# tensor* division(tensor* t0, tensor* t1, bool b);
lib.division.argtypes = [ctypes.POINTER(Tensor), ctypes.POINTER(Tensor), ctypes.c_bool]
lib.division.restype  = ctypes.POINTER(Tensor)

# tensor* add(tensor* t0, tensor* t1, bool retain_grad);
lib.add.argtypes = [ctypes.POINTER(Tensor), ctypes.POINTER(Tensor), ctypes.c_bool]
lib.add.restype  = ctypes.POINTER(Tensor)

# tensor* relu(tensor* t0, bool retain_grad);
lib.relu.argtypes = [ctypes.POINTER(Tensor), ctypes.c_bool]
lib.relu.restype  = ctypes.POINTER(Tensor)

# tensor* exponential(tensor* t0, bool retain_grad);
lib.exponential.argtypes = [ctypes.POINTER(Tensor), ctypes.c_bool]
lib.exponential.restype  = ctypes.POINTER(Tensor)

# tensor* power(tensor* t0, tensor* t1, bool retain_grad);
lib.power.argtypes = [ctypes.POINTER(Tensor), ctypes.POINTER(Tensor), ctypes.c_bool]
lib.power.restype  = ctypes.POINTER(Tensor)

# tensor* sigmoid(tensor* t0, bool retain_grad);
lib.sigmoid.argtypes = [ctypes.POINTER(Tensor), ctypes.c_bool]
lib.sigmoid.restype  = ctypes.POINTER(Tensor)

# tensor* logarithm(tensor* t0, bool retain_grad);
lib.logarithm.argtypes = [ctypes.POINTER(Tensor), ctypes.c_bool]
lib.logarithm.restype  = ctypes.POINTER(Tensor)

# tensor* neg(tensor* t0, bool retain_grad);
lib.neg.argtypes = [ctypes.POINTER(Tensor), ctypes.c_bool]
lib.neg.restype  = ctypes.POINTER(Tensor)

# tensor* square_root(tensor* t0, bool retain_grad);
lib.square_root.argtypes = [ctypes.POINTER(Tensor), ctypes.c_bool]
lib.square_root.restype  = ctypes.POINTER(Tensor)

# tensor* abs(tensor* t0, bool retain_grad);
lib.abs.argtypes = [ctypes.POINTER(Tensor), ctypes.c_bool]
lib.abs.restype  = ctypes.POINTER(Tensor)

# tensor* sign(tensor* t0, bool retain_grad);
lib.sign.argtypes = [ctypes.POINTER(Tensor), ctypes.c_bool]
lib.sign.restype  = ctypes.POINTER(Tensor)

# tensor* reciprocal(tensor* t0, bool retain_grad);
lib.reciprocal.argtypes = [ctypes.POINTER(Tensor), ctypes.c_bool]
lib.reciprocal.restype  = ctypes.POINTER(Tensor)

#tensor * sum(tensor *t0, tensor *dim_data, bool retain_grad)
lib.sum.argtypes = [ctypes.POINTER(Tensor), ctypes.POINTER(Tensor), ctypes.c_bool]
lib.sum.restype  = ctypes.POINTER(Tensor)

#tensor * view(tensor *t0, tensor *dim_data)
lib.view.argtypes = [ctypes.POINTER(Tensor), ctypes.POINTER(Tensor)]
lib.view.restype  = ctypes.POINTER(Tensor)

#tensor * expand(tensor *t0, tensor *dim_data)
lib.expand.argtypes = [ctypes.POINTER(Tensor), ctypes.POINTER(Tensor)]
lib.expand.restype  = ctypes.POINTER(Tensor)

#tensor * permute(tensor *t0, tensor *dim_data)
lib.permute.argtypes = [ctypes.POINTER(Tensor), ctypes.POINTER(Tensor)]
lib.permute.restype  = ctypes.POINTER(Tensor)

#tensor * bmm(tensor *t0, tensor *t1, bool retain_grad)
lib.bmm.argtypes = [ctypes.POINTER(Tensor), ctypes.POINTER(Tensor), ctypes.c_bool]
lib.bmm.restype  = ctypes.POINTER(Tensor)

#tensor * bcmm(tensor *t0, tensor *t1, bool retain_grad)
lib.bcmm.argtypes = [ctypes.POINTER(Tensor), ctypes.POINTER(Tensor), ctypes.c_bool]
lib.bcmm.restype  = ctypes.POINTER(Tensor)

#tensor *isclose(tensor *a, tensor *b, lemur_float rtol, lemur_float atol){
lib.isclose.argtypes = [ctypes.POINTER(Tensor), ctypes.POINTER(Tensor), ctypes.c_float,  ctypes.c_float]
lib.isclose.restype = ctypes.POINTER(Tensor)