from typing import Optional
import ctypes
from frontend.bindings import lib, lemur_float
import frontend.reprutils as reprutils
import weakref

class LemurTensor:
    __slots__ = ("_ptr", "_parents", "__weakref__")
    #TODO make note that _parents is needed so that when doing w = w.relu() or similar, GC doesnt mess us up

    def __init__(self, 
             shape: Optional[list[int]] = None, 
             requires_grad: Optional[bool] = False, 
             _ptr = None, 
             _parents = None):
        
        if _ptr is not None:
            self._ptr = _ptr
            if self.requires_grad():
                self._parents = tuple(p for p in _parents)
            else:
                self._parents = tuple(weakref.ref(p) for p in _parents) if _parents else ()

        else:
            self._parents = tuple()
            if shape is None or shape[-1] is None:
                shape = (1,)
            c_shape = (ctypes.c_size_t * 5)(*([1]*5))
            for i, dim in enumerate(shape):
                c_shape[i] = dim
            retains_grad = requires_grad #because created by user.
            t_ptr = lib.empty_tensor(c_shape, requires_grad, retains_grad)
            if not t_ptr:
                raise RuntimeError("empty_tensor returned NULL.")
            self._ptr = t_ptr

    ### helpers ###
    @staticmethod
    def _convert_to_tensor(obj):
        if isinstance(obj, (tuple, list)):
            obj = tensor(obj)
        if not isinstance(obj, LemurTensor):
            raise TypeError("Input must be a LemurTensor, tuple, or list.")
        return obj
    
    def _process_args(self, *args):
        if len(args) == 1 and isinstance(args[0], (tuple, list, LemurTensor)):
            return self._convert_to_tensor(args[0])
        return self._convert_to_tensor(list(args))
    
    ### garbage collection ###
    def __del__(self):
        if getattr(self, "_ptr", None) is not None:
            lib.free_tensor(self._ptr)
            self._ptr = None

    def detach(self):
        # manually detaches parent references to allow garbage collection.
        self._parents = ()
    
    @property
    def parents(self):
        if self.requires_grad:
            return self._parents
        else:
            #  weak references, so dereference them
            return tuple(ref() for ref in self._parents)
    
    ### print ###
    def __repr__(self):
        return reprutils._tensor_repr(self._ptr)
    
    ### properties/utility methods ###
    @staticmethod
    def _contiguous_deepcopy_k(kt_ptr):
        if ctypes.cast(kt_ptr, ctypes.c_void_p).value is None:
            raise ValueError("Attempting to call _contiguous_deepcopy_k on NULL kernel tensor")
            return None
        return LemurTensor(_ptr=lib.tensor_from(lib.contiguous_deepcopy_kernel_tensor(kt_ptr), None, None, None))

    def __getitem__(self, index): #Add slicing  
        if index >= self.memory_length:
            raise ValueError("Invalid memory access.")
            return None
        else:
            return float(self._ptr.contents.k.contents.array[index].value)
        
    def __setitem__(self, index, value):
        if index >= self.memory_length:
            raise ValueError("Invalid memory access.")
        else:
            self._ptr.contents.k.contents.array[index] = lemur_float(value)

    @property
    def grad(self):
        if ctypes.cast(self._ptr.contents.grad, ctypes.c_void_p).value is None:
            return None
        return self._contiguous_deepcopy_k(self._ptr.contents.grad)
    
    def requires_grad(self):
        return self._ptr.contents.requires_grad
    
    @property
    def graph(self):
        return reprutils.plot_tensor_graph_parents(self)
    
    def stride(self):
        return tensor([self._ptr.contents.k.contents.stride[i] for i in range(5)])
    
    def is_shallow(self):
        return self._ptr.contents.k.contents.shallow

    def is_contiguous(self):
        return lib.is_contiguous(self._ptr.contents.k)
        
    @property
    def memory_length(self):
        return self._ptr.contents.k.contents.length
    
    @property
    def shape(self):
        return tensor([self._ptr.contents.k.contents.shape[i] for i in range(5)])

    def numel(self):
        shape = self.shape
        return shape[0] * shape[1] * shape[2] * shape[3] * shape[4]
    
    def ndimension(self):  
        return 5
    
    ### grad/compile ops###
    def backward(self):
        lib.backward(self._ptr)

    def compile(self):
        lib.compile(self._ptr)

    ### Binary ops ###
    def __add__(self, other):
        if not isinstance(other, LemurTensor):
            raise TypeError("Can't add LemurTensor with non-LemurTensor.")
        c_result = lib.add(self._ptr, other._ptr, False)
        return LemurTensor(_ptr=c_result, _parents=(self, other))
    
    def __sub__(self, other):
        if not isinstance(other, LemurTensor):
            raise TypeError("Can't subtract LemurTensor with non-LemurTensor.")
        c_result = lib.sub(self._ptr, other._ptr, False)
        return LemurTensor(_ptr=c_result, _parents=(self, other))

    def __mul__(self, other):
        if not isinstance(other, LemurTensor):
            raise TypeError("Can't multiply LemurTensor with non-LemurTensor.")
        c_result = lib.mul(self._ptr, other._ptr, False)
        return LemurTensor(_ptr=c_result, _parents=(self, other))
    
    def __truediv__(self, other):
        if not isinstance(other, LemurTensor):
            raise TypeError("Can't divide LemurTensor with non-LemurTensor.")
        c_result = lib.division(self._ptr, other._ptr, False)
        return LemurTensor(_ptr=c_result, _parents=(self, other))
    
    def __eq__(self, other):
        if not isinstance(other, LemurTensor):
            raise TypeError("Can't divide LemurTensor with non-LemurTensor.")
        c_result = lib.eq(self._ptr, other._ptr)
        return LemurTensor(_ptr=c_result, _parents=(self, other))

    ### Reduce ops ###
    def sum(self, *args):
        if not args: 
            dims = [0,0,0,0,0]
        else:
            dims = [1,1,1,1,1]
        for d in args:
            dims[d] = 0
        other = self._convert_to_tensor(dims)
        c_result = lib.sum(self._ptr, other._ptr, False)
        return LemurTensor(_ptr=c_result, _parents=(self,other))
    
    def all(self, *args):
        if not args: 
            dims = [0,0,0,0,0]
        else:
            dims = [1,1,1,1,1]
        for d in args:
            dims[d] = 0
        other = self._convert_to_tensor(dims)
        c_result = lib.all(self._ptr, other._ptr, False)
        return LemurTensor(_ptr=c_result, _parents=(self,other))
    
    def any(self, *args):
        if not args: 
            dims = [0,0,0,0,0]
        else:
            dims = [1,1,1,1,1]
        for d in args:
            dims[d] = 0
        other = self._convert_to_tensor(dims)
        c_result = lib.any(self._ptr, other._ptr, False)
        return LemurTensor(_ptr=c_result, _parents=(self,other))
    
    ### Unary ops ###
    def __pow__(self, other):
        if not isinstance(other, LemurTensor):
            if isinstance(other, float) or isinstance(other, int):
                other = tensor([float(other)])
            else:
                raise TypeError("Can't take LemurTensor to non-float or non-LemurTensor exponent.")
        c_result = lib.power(self._ptr, other._ptr, False)
        return LemurTensor(_ptr=c_result, _parents=(self, other))
    
    def exp(self):
        c_result = lib.exponential(self._ptr, False)
        return LemurTensor(_ptr=c_result, _parents=(self,))
    
    def relu(self):
        c_result = lib.relu(self._ptr, False)
        return LemurTensor(_ptr=c_result, _parents=(self,))
    
    def sigmoid(self):
        c_result = lib.sigmoid(self._ptr, False)
        return LemurTensor(_ptr=c_result, _parents=(self,))
            
    def log(self):
        c_result = lib.logarithm(self._ptr, False)
        return LemurTensor(_ptr=c_result, _parents=(self,))
    
    def neg(self):
        c_result = lib.neg(self._ptr, False)
        return LemurTensor(_ptr=c_result, _parents=(self,))
    
    def sqrt(self):
        c_result = lib.square_root(self._ptr, False)
        return LemurTensor(_ptr=c_result, _parents=(self,))
    
    def abs(self):
        c_result = lib.abs(self._ptr, False)
        return LemurTensor(_ptr=c_result, _parents=(self,))
    
    def sign(self):
        c_result = lib.sign(self._ptr, False)
        return LemurTensor(_ptr=c_result, _parents=(self,))
    
    def reciprocal(self):
        c_result = lib.reciprocal(self._ptr, False)
        return LemurTensor(_ptr=c_result, _parents=(self,))
    
    ### Shape ops ###
    def flatten(self, dim=4):
        total_elements = self.numel()
        view_dim = [1,1,1,1,1]
        view_dim[dim] = total_elements
        return self.view(view_dim)
    
    def view(self, *args):
        other = self._process_args(*args)
        c_result = lib.view(self._ptr, other._ptr)
        return LemurTensor(_ptr=c_result, _parents=(self, other)) #TODO: maybe other should not be in shape ops

    def expand(self, *args):
        other = self._process_args(*args)
        c_result = lib.expand(self._ptr, other._ptr)
        return LemurTensor(_ptr=c_result, _parents=(self, other))

    def permute(self, *args):
        other = self._process_args(*args)
        c_result = lib.permute(self._ptr, other._ptr)
        return LemurTensor(_ptr=c_result, _parents=(self, other))

    ### matmul ###
    def __matmul__(self, other):
        if (other.shape[0] == 1 and other.shape[1] == 1 and other.shape[2] == 1):
            c_result = lib.bcmm(self._ptr, other._ptr, False)
        else:
            c_result = lib.bmm(self._ptr, other._ptr, False)
        return LemurTensor(_ptr=c_result, _parents=(self, other))


def empty(shape, requires_grad=False):
    t = LemurTensor(shape=shape, requires_grad=requires_grad)
    return t

def _infer_shape(data):
    if not isinstance(data, list):
        return []
    
    if len(data) == 0:
        return [0]
    
    first_sub_shape = _infer_shape(data[0])

    top_shape = [len(data)] + first_sub_shape
    
    for sub in data[1:]:
        sub_shape = _infer_shape(sub)
        if sub_shape != first_sub_shape:
            raise ValueError("Inconsistent dimensions encountered in nested list.")
    return top_shape

def _flatten_data(data):
    if not isinstance(data, list):
        return [data]
    flat = []
    for sub in data:
        flat.extend(_flatten_data(sub))
    return flat

def tensor(data, requires_grad=False):
    
    inferred_shape = _infer_shape(data)  # e.g. [2, 3, 4]
    if len(inferred_shape) > 5:
        raise ValueError("Data has more than 5 dimensions, which is not supported.")
    
    pad_length = 5 - len(inferred_shape)
    final_shape = [1] * pad_length + inferred_shape 
    
    t = empty(shape=final_shape, requires_grad=requires_grad)

    flat_data = _flatten_data(data)

    k_ptr = t._ptr.contents.k
    c_arr = k_ptr.contents.array

    if len(flat_data) != (final_shape[0] * 
                          final_shape[1] * 
                          final_shape[2] *
                          final_shape[3] *
                          final_shape[4]):
        raise ValueError("Number of elements in `data` does not match the tensor's shape.")
    
    for i, val in enumerate(flat_data):
        c_arr[i] = val

    return t
