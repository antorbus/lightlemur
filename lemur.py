from __future__ import annotations
import numpy as np
from typing import final, Callable, TypeVar, NamedTuple
from dataclasses import dataclass

#TODO decide on STYLE (camelCase?)

#NOTE When creating backwards and forwards kernels for unary, binary and reduce ops,
#please create the result contiguously in memory. This is extremely important as
#lightlemur assumes that you have done this.

T = TypeVar('T')
NestedList = T | list['NestedList[T]']

class OperationMeta(type):
    def __new__(cls, name, bases, dct):
        def has_required_methods(class_dict):
            required_methods = ['forward', 'backward', 'forward_kernel', 'backward_kernel']
            return all(any(callable(getattr(base, meth, None)) or meth in class_dict for base in bases + (class_obj,))
                       for meth in required_methods)
        
        class_obj = super().__new__(cls, name, bases, dct)
        
        if not has_required_methods(dct):
            missing_methods = [
                meth for meth in ['forward', 'backward', 'forward_kernel', 'backward_kernel']
                if not any(callable(getattr(base, meth, None)) or meth in dct for base in bases + (class_obj,))
            ]
            raise AttributeError(f"Class {name} is missing required methods: {', '.join(missing_methods)}")
        
        return class_obj
    
class UnaryOperation(metaclass=OperationMeta):
    '''
    Element-wise functions which have one tensor as input and one of the same shape as output. 
    '''
    
    #@classmethod
    #def __call__(): TODO: create this
    
    #TODO WHEN NOT USING NUMPY NEED TO MAKE A COPY OF data
    
    @final
    @classmethod
    def forward(cls, a: Tensor, _retain_grad : bool = False) -> Tensor:
        data = cls.forward_kernel(a.data)
        assert len(data) == len(a.data), "Unary operation must perserve shape."
        return Tensor(data = data,
                      shape = a.shape,
                      requires_grad = a._requires_grad,
                      dtype = a.dtype,
                      comes_from  = Expression(tuple(a), cls.backward),
                      _retain_grad = _retain_grad)        
        
    @final
    @classmethod
    def backward(cls, t : tuple[Tensor], seed : np.ndarray,
                 seed_shape : tuple[int], seed_stride : tuple[int]) -> None:
        #Since unary, there will only be one element in t 
        derive(t[0], *cls.backward_kernel(t[0].data, seed))
        
    @staticmethod
    def forward_kernel(a : np.ndarray, 
                       shape : tuple[int],
                       stride: tuple[int]) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement 'forward_kernel' function")
   
    @staticmethod
    def backward_kernel(a : np.ndarray, seed : np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement 'backward_kernel' function")


class relu(UnaryOperation):
    @staticmethod
    def forward_kernel(a : np.ndarray) -> np.ndarray:
        return a * (a > 0)  

    @staticmethod
    def backward_kernel(a: np.ndarray, seed : np.ndarray) -> np.ndarray: #returns next seed        
        next_seed = seed
        next_seed[a < 0]  = 0
        return next_seed

class sigmoid(UnaryOperation):
    @staticmethod
    def forward_kernel(a : np.ndarray) -> np.ndarray:
        return 1/(1 + np.exp(-a))

    @staticmethod
    def backward_kernel(a: np.ndarray, seed : np.ndarray) -> np.ndarray: 
        sigmoid_input = sigmoid.forward_kernel(a)
        return seed * sigmoid_input * (1 - sigmoid_input)

    

class BinaryOperation(metaclass=OperationMeta):
    '''
    Functions which have two tensors of the same shape as input and one as 
    output of the same shape. 
    '''
    @final
    @classmethod
    def forward(cls, a: Tensor, b: Tensor, _retain_grad : bool = False) -> Tensor:
        assert a.shape == b.shape, "The shape of the inputs must match for a binary operation."
        data = cls.forward_kernel(a.data, b.data, a.stride, b.stride)
        assert len(data) == len(a.data), "Binary operation must perserve shape."
        return Tensor(data = data,
                      shape = a.shape,
                      requires_grad = a._requires_grad or b._requires_grad,
                      dtype = a.dtype,
                      comes_from  = Expression(tuple(a,b),cls.backward),
                      _retain_grad = _retain_grad)        
        
    @final
    @classmethod
    def backward(cls, t : list[Tensor], seed : np.ndarray,
                 seed_shape : tuple[int], seed_stride : tuple[int]) -> None:
        #TODO: This can probably be multithread 
        derive(t[0], *cls.backward_kernel(t.data, seed, 0))
        derive(t[1], *cls.backward_kernel(t.data, seed, 1))
        
    @staticmethod
    def forward_kernel(a : np.ndarray, b : np.ndarray,
                       seed_shape : tuple[int], seed_stride : tuple[int]) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement 'forward_kernel' function")
   
    @staticmethod
    def backward_kernel(t: tuple[np.ndarray], seed : np.ndarray, idx : int) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement 'backward_kernel' function")


class addTensors(BinaryOperation):
    @staticmethod
    def forward_kernel(a : np.ndarray, b : np.ndarray) -> np.ndarray:
        return a + b, a.shape
   
    @staticmethod
    def backward_kernel(t: list[np.ndarray], seed : np.ndarray, idx : int) -> np.ndarray:
        return seed

class hadamardProduct(BinaryOperation):
    @staticmethod
    def forward_kernel(a : np.ndarray, b : np.ndarray) -> np.ndarray:
        return a * b
   
    @staticmethod
    def backward_kernel(t: list[np.ndarray], seed : np.ndarray, idx : int) -> np.ndarray:
        if idx == 0:
            return seed * t[1]
        if idx == 1:
            return seed * t[0]
      



class ReduceOperation(metaclass=OperationMeta):
    '''
    Functions which have one tensors and a dimension as input, the output
    is a tensor with one dimension less. 
    '''
    @final
    @classmethod
    def forward(cls, a : Tensor, dim : int, _retain_grad : bool = False) -> Tensor:
        assert dim < a.dim, f"Cannot reduce along dimension {dim} when tensor.dim is {a.dim}"
        shape = list(a.shape)
        del shape[dim]
        return Tensor(data = cls.forward_kernel(a.data, dim, a.shape ),
                      shape = tuple(shape),
                      requires_grad = a._requires_grad,
                      dtype = a.dtype,
                      comes_from  = Expression(tuple(a),cls.backward),
                      _retain_grad = _retain_grad)        
        
    @final
    @classmethod
    def backward(cls, t : tuple[Tensor], dim : int, seed : np.ndarray,
                 seed_shape : tuple[int], seed_stride : tuple[int]) -> None:
        derive(t[0], *cls.backward_kernel(t[0].data, dim,  seed))
        
    @staticmethod
    def forward_kernel(a : Tensor, dim : int) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement 'forward_kernel' function")
   
    @staticmethod
    def backward_kernel(t: Tensor, dim: int, seed : np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement 'backward_kernel' function")

class sum(ReduceOperation):    
    @staticmethod
    def forward_kernel(t : np.ndarray, dim : int, shape : tuple[int], stride) -> np.ndarray:
        return t.mean()
   
    @staticmethod
    def backward_kernel(t: Tensor, seed : np.ndarray) -> np.ndarray:
        #seed must be 1d TODO
        n_elements = t.numel
        next_seed = np.ones((t.numel,), dtype = t.dtype) * (seed/n_elements)
        return next_seed


class ShapeOperation(metaclass=OperationMeta):
    '''
    Functions which have one tensors as input and change the shape and stride of it. 
    '''
    @final
    @classmethod
    def forward(cls, a : Tensor, _retain_grad : bool = False) -> Tensor:
        shape, stride = cls.forward_kernel(a.shape, a.stride)
        return Tensor(data = a.data.copy(),
                      shape = shape,
                      stride = stride,
                      requires_grad = a._requires_grad,
                      dtype = a.dtype,
                      comes_from  = Expression([a], cls.backward),
                      _retain_grad = _retain_grad)        
        
    @final
    @classmethod
    def backward(cls, t : tuple[Tensor], seed : np.ndarray,
                 seed_shape : tuple[int], seed_stride : tuple[int]) -> None:
        derive(t[0], seed, *cls.backward_kernel(t[0].shape, t[0].stride, seed_shape, seed_stride))
        
    
    @staticmethod
    def forward_kernel(shape : tuple[int], stride : tuple[int]) -> tuple[tuple[int], tuple[int]]:
        raise NotImplementedError("Subclasses must implement 'forward_kernel' function")
   
    @staticmethod
    def backward_kernel(input_shape : tuple[int], 
                        input_stride : tuple[int],
                        seed_shape : tuple[int], 
                        seed_stride : tuple[int]) -> tuple[tuple[int], tuple[int]]:
        raise NotImplementedError("Subclasses must implement 'backward_kernel' function")               
        

#TODO COMPOSITE OPERATION CLASS

def matmul(a : Tensor, b : Tensor) -> Tensor:
    assert a.dim == 2, "Tensor must be 2 dimensional"
    assert b.dim == 2, "Tensor must be 2 dimensional"
    
    assert a.shape[-1] == b.shape[-2], f"Size mismatch, cannot multiply tensors of sizes {a.shape} {b.shape}"
    #A shape is i j
    #B shape is j k

    #B = Transpose B
    #B shape is k j
    # 
    #expand A to i*k j
    #expand B to k*i j
    #C = hadamard product i*k j
    #reduce by sum across j
    #view C as i k
    #win
     


@dataclass(frozen=True)
class Expression:
    inputs: tuple[Tensor]
    backward: Callable


def derive(a : Tensor, 
           seed : np.ndarray, 
           seed_shape : tuple[int], 
           seed_stride : tuple[int]) -> None:

    if a._partial_array is not None:
        a._partial_array += seed

    if a.comes_from is not None:
         a.comes_from.backward(a.comes_from.inputs, seed, seed_shape, seed_stride)


class Tensor(): 
    """
    If data is None then one or both of shape and _array will be used.
    If only shape is passed, a tensor of that shape will be created with uninitialized values.
    If only _array is passed, a 1d tensor will be created with shape [len(_array)].
    If shape and stride are passed, the stride will be validated.
    """

    def __init__(self, 
                 data : NestedList[ int | float | bool ] | np.ndarray,
                 shape : tuple[int] = None,
                 stride : tuple[int] = None,
                 requires_grad : bool = False, 
                 dtype : type  = float, 
                 comes_from : Expression = None,
                 _retain_grad : bool = None,
                ):
        
        
        self.data  = np.array(data, dtype=dtype).flatten()
                 
        if shape is not None: 
            self.shape = shape
        elif type(data) == np.ndarray:
            self.shape = data.shape 
        else:
            self.shape = self._get_shape(data)
        
        if stride is not None:
            self.stride = stride
        else:
            self.stride = self.get_contiguous_stride()
        
        assert self.numel == len(self.data), ValueError("Dimensions must be regular")
        
        assert self.is_stride_shape_compatible(), "Stride not compatible with shape"
        
        self._requires_grad = requires_grad
        
        if _retain_grad is None: #This means tensor created by user
            self._retain_grad = self._requires_grad 
        else: #This means the tensor is created by a function 
            self._retain_grad = _retain_grad 
        
        if self._retain_grad:
            assert self._requires_grad == True, "_retain_grad can only be True is requires_grad is True"
            self._partial_array = np.full(self.numel, 0, dtype=dtype)
        else:
            self._partial_array = None
        
        self.comes_from = comes_from

        assert type(self.shape) == tuple, "Shape must be immutable"
        assert type(self.stride) == tuple, "Stride must be immutable"

        self.dtype = dtype

   

    def get_contiguous_stride(self):
        stride = [1]
        for s in self.shape[-1:0:-1]:  
            stride.append(stride[-1] * s)
        return tuple(reversed(stride))
        
    def is_stride_shape_compatible(self):
        if len(self.shape) != len(self.stride):
            return False
        
        if any(s < 0 for s in self.stride):
            return False
        
        #TODO think about if expand should be included
        if any(size > 0 and stride == 0 for size, stride in zip(self.shape, self.stride)):
            return False
        
        seen = set()
        def check_overlap(dim, offset):
            if dim == len(self.shape):
                if offset in seen:
                    return False
                seen.add(offset)
                return True
                
            for i in range(self.shape[dim]):
                if not check_overlap(dim + 1, offset + i * self.stride[dim]):
                    return False
            return True
        return check_overlap(0, 0)
    
    @classmethod
    def _get_shape(self,lst):
        def _get_size( lst):
            if isinstance(lst, list):
                return [len(lst)] + _get_size(lst[0]) if len(lst) > 0 else [0]
            else:
                return []
        return tuple(_get_size( lst))

    def __repr__(self):
        def format_array(array, shape, strides, base_offset=0):
            if len(shape) == 0:
                return '[]'
            elif len(shape) == 1:
                elements = [str(array[base_offset + i * strides[0]]) 
                        for i in range(shape[0])]
                return "[" + ", ".join(elements) + "]"
            else:
                result = []
                for i in range(shape[0]):
                    new_offset = base_offset + i * strides[0]
                    sub_array = format_array(array, 
                                        shape[1:], 
                                        strides[1:], 
                                        new_offset)
                    result.append(sub_array)
                return "[" + ",\n ".join(result) + "]"
            
        string = format_array(self.data, self.shape, self.stride) + \
                f"\nShape: {self.shape}\nStride: {self.stride}\nDtype: {self.dtype}"
        
        if self._requires_grad:
            string += f"\nBackward: "
            if self.comes_from is not None:
                string += f"{self.comes_from.backward}"
            else:
                string += "None"
                
            if self._retain_grad:
                string += "\n" + format_array(self._partial_array, 
                                            self.shape, 
                                            self.stride)
                
        return string

    def retain_grad(self, set_ret_grad : bool):
        
        if set_ret_grad == True and self._retain_grad == True:
            return #we do not want to overwrite _partial_array
        
        self._retain_grad = set_ret_grad 
        if set_ret_grad:
            self.requires_grad(True)
            self._partial_array = np.full(self.numel, 0, dtype=self.dtype)
        else:
            self._partial_array = None
        
    def requires_grad(self, set_req_grad : bool):
        self._requires_grad = set_req_grad
        if self._requires_grad == False:
            self._retain_grad = False
            self._partial_array = None 
        
    @property
    def numel(self):
        if self.shape:
            result = 1  
            for num in self.shape:
                result *= num
            return result
        else: 
            return 0
    
    @property
    def dim(self):
        return len(self.shape)
    
    def backward(self):
        assert self.numel == 1, "Backward is only defined when Tensor is a scalar"
        seed = np.array([1],dtype = self.dtype)
        derive(self, seed)


    
    


    





