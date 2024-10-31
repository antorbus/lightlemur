from __future__ import annotations
from typing import final, Callable, TypeVar, NamedTuple
from dataclasses import dataclass
import math 
#TODO decide on STYLE (camelCase?) PEP

#NOTE All tensors are contigous, so stride is always what you would expect/
#NOTE backwards kernel seed tensors cannot be modified inplace!
#NOTE forward kernel tensor are NOT inplace 

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

        k = cls.forward_kernel(a._k)

        assert k.shape == a.shape, "Unary operation must perserve shape."
        assert k.dtype == a.dtype, "Unary operation must perserve dtype."
        assert k.memory_length == a._k.memory_length, "Unary operation must perserve memory length."
        
        return Tensor(array = None,
                      shape = None,
                      requires_grad = a._requires_grad,
                      dtype = None,
                      comes_from  = Expression(tuple(a), cls.backward),
                      _retain_grad = _retain_grad,
                      _k = k)        
        
    @final
    @classmethod
    def backward(cls, t : tuple[Tensor], seed : KernelTensor) -> None:
        #Since unary, there will only be one element in t 
        derive(t[0], cls.backward_kernel(t[0]._k, seed))
        
    @staticmethod
    def forward_kernel(a : KernelTensor) -> KernelTensor:
        raise NotImplementedError("Subclasses must implement 'forward_kernel' function")
   
    @staticmethod
    def backward_kernel(a : KernelTensor, seed : KernelTensor) -> KernelTensor:
        raise NotImplementedError("Subclasses must implement 'backward_kernel' function")


class relu(UnaryOperation):
    @staticmethod
    def forward_kernel(a : KernelTensor) -> KernelTensor:
        arr = [None]*a.memory_length
        for idx in range(a.memory_length):
            arr[idx] = a.array[idx] if a.array[idx] > 0 else 0.
        
        return KernelTensor(array= tuple(arr),
                            memory_length=a.memory_length,
                            shape=a.shape,
                            dtype=a.dtype)

    @staticmethod
    def backward_kernel(a: KernelTensor, seed : KernelTensor) -> KernelTensor:  
        next_seed_arr = [None]*seed.memory_length
        for idx in range(seed.memory_length):
            next_seed_arr[idx] = seed.array[idx] if a.array[idx] > 0 else 0. 
        return KernelTensor(array= tuple(next_seed_arr),
                            memory_length=seed.memory_length,
                            shape=seed.shape,
                            dtype=seed.dtype)      
 

class sigmoid(UnaryOperation):
    @staticmethod
    def forward_kernel(a : KernelTensor) -> KernelTensor:
        arr = [None]*a.memory_length
        for idx in range(a.memory_length):
            arr[idx] = 1./(1.+math.exp(-1.*a.array[idx])) 
        
        return KernelTensor(array= tuple(arr),
                            memory_length=a.memory_length,
                            shape=a.shape,
                            dtype=a.dtype)
        # return 1/(1 + np.exp(-a))

    @staticmethod
    def backward_kernel(a: KernelTensor, seed : KernelTensor) -> KernelTensor: 

        next_seed_arr = [None]*seed.memory_length
        for idx in range(seed.memory_length):
            element_sigmoid = 1./(1.+math.exp(-1.*a.array[idx])) 
            next_seed_arr[idx] = seed.array[idx] * element_sigmoid * (1. - element_sigmoid)

        return KernelTensor(array= tuple(next_seed_arr),
                            memory_length=seed.memory_length,
                            shape=seed.shape,
                            dtype=seed.dtype)  
        # sigmoid_input = sigmoid.forward_kernel(a)
        # return seed * sigmoid_input * (1 - sigmoid_input)

    

class BinaryOperation(metaclass=OperationMeta):
    '''
    Functions which have two tensors of the same shape as input and one as 
    output of the same shape. 
    '''
    @final
    @classmethod
    def forward(cls, a: Tensor, b: Tensor, _retain_grad : bool = False) -> Tensor:

        assert a.shape == b.shape, "The shape of the inputs must match for a binary operation."
        assert a.dtype == b.dtype, "The type of the inputs must match for a binary operation."
        assert a._k.memory_length == b._k.memory_length, "Memory length mismatch."

        k = cls.forward_kernel(a._k, b._k)
        
        assert k.shape == a.shape, "Binary operation must perserve shape."
        assert k.dtype == a.dtype, "Binary operation must perserve dtype."
        assert k.memory_length == a._k.memory_length, "Binary operation must perserve memory length."

        return Tensor(data = None,
                      shape = None,
                      requires_grad = a._requires_grad or b._requires_grad,
                      dtype = None,
                      comes_from  = Expression(tuple(a,b),cls.backward),
                      _retain_grad = _retain_grad,
                      _k = k)   
   
        
    @final
    @classmethod
    def backward(cls, t : tuple[Tensor], seed : KernelTensor,) -> None:
        #TODO: This can probably be multithread 
        derive(t[0], cls.backward_kernel((t[0]._k, t[1]._k), seed, 0))
        derive(t[1], cls.backward_kernel((t[0]._k, t[1]._k), seed, 1))
        
    @staticmethod
    def forward_kernel(a : KernelTensor, b : KernelTensor) -> KernelTensor:
        raise NotImplementedError("Subclasses must implement 'forward_kernel' function")
   
    @staticmethod
    def backward_kernel(t: tuple[KernelTensor], seed : KernelTensor, idx : int) -> KernelTensor:
        raise NotImplementedError("Subclasses must implement 'backward_kernel' function")


class addTensors(BinaryOperation):
    @staticmethod
    def forward_kernel(a : KernelTensor, b : KernelTensor) -> KernelTensor:
        arr = [None]*a.memory_length
        for idx in range(a.memory_length):
            arr[idx] = a.array[idx]+b.array[idx]
        
        return KernelTensor(array= tuple(arr),
                            memory_length=a.memory_length,
                            shape=a.shape,
                            dtype=a.dtype)
        #return a + b
   
    @staticmethod
    def backward_kernel(t: tuple[KernelTensor], seed : KernelTensor, idx : int) -> KernelTensor:
        next_seed_arr = [None]*seed.memory_length
        for idx in range(seed.memory_length):
            next_seed_arr[idx] = seed.array[idx]

        return KernelTensor(array= tuple(next_seed_arr),
                            memory_length=seed.memory_length,
                            shape=seed.shape,
                            dtype=seed.dtype)  

class hadamardProduct(BinaryOperation):
    @staticmethod
    def forward_kernel(a : KernelTensor, b : KernelTensor) -> KernelTensor:
        arr = [None]*a.memory_length
        for idx in range(a.memory_length):
            arr[idx] = a.array[idx]*b.array[idx]
        
        return KernelTensor(array= tuple(arr),
                            memory_length=a.memory_length,
                            shape=a.shape,
                            dtype=a.dtype)
        #return a * b
   
    @staticmethod
    def backward_kernel(t: tuple[KernelTensor], seed : KernelTensor, idx : int) -> KernelTensor:
        if idx == 1:
            t_idx = 0
        else: #idx ==0 
            t_idx = 1
        next_seed_arr = [None]*seed.memory_length
        for idx in range(seed.memory_length):
            next_seed_arr[idx] = seed.array[idx]*t[t_idx].array[idx]
    
        return KernelTensor(array= tuple(next_seed_arr),
                            memory_length=seed.memory_length,
                            shape=seed.shape,
                            dtype=seed.dtype) 
        # if idx == 0:  
        #     return seed * t[1]
        # if idx == 1:
        #     return seed * t[0]
      



class ReduceOperation(metaclass=OperationMeta):
    '''
    Functions which have one tensors and a dimension as input, the output
    is a tensor with one dimension less. 
    '''
    @final
    @classmethod
    def forward(cls, a : Tensor, dim : int, _retain_grad : bool = False) -> Tensor:

        assert dim < a.dim, f"Cannot reduce along dimension {dim} when tensor.dim is {a.dim}"

        k = cls.forward_kernel(a._k, dim)
        assert k.dtype == a.dtype, "Reduce operation must perserve dtype."
        assert len(k.shape) == len(a.shape) -1, "Reduce did not remove one dimension."

        #TODO check memory length reduce appropiately

        proper_reduce = False
        for i in range(len(a.shape)):
            if a.shape[:i] + a.shape[i+1:] == k.shape:
                proper_reduce = True

        assert proper_reduce == True, "Dimensions changed unexpectedly after reduce."

        return Tensor(data = None,
                      shape = None,
                      requires_grad = a._requires_grad,
                      dtype = None,
                      comes_from  = Expression(tuple(a),cls.backward),
                      _retain_grad = _retain_grad,
                      _k = k)     

        
    @final
    @classmethod
    def backward(cls, t : tuple[Tensor], dim : int, seed : KernelTensor) -> None:
        derive(t[0], cls.backward_kernel(t[0]._k, dim, seed))
        
    @staticmethod
    def forward_kernel(a : KernelTensor, dim : int) -> KernelTensor:
        raise NotImplementedError("Subclasses must implement 'forward_kernel' function")
   
    @staticmethod
    def backward_kernel(t: KernelTensor, dim: int, seed : KernelTensor) -> KernelTensor:
        raise NotImplementedError("Subclasses must implement 'backward_kernel' function")

class sum(ReduceOperation):    
    @staticmethod
    def forward_kernel(t : KernelTensor, dim : int) -> KernelTensor:
        pass
        #return t.mean()
   
    @staticmethod
    def backward_kernel(t: KernelTensor, dim: int, seed : KernelTensor) -> KernelTensor:
        pass
        #seed must be 1d TODO
        # n_elements = t.numel
        # next_seed = np.ones((t.numel,), dtype = t.dtype) * (seed/n_elements)
        # return next_seed


#TODO NEED TO GIVE MORE THOUGHT
class ShapeOperation(metaclass=OperationMeta):
    '''
    Functions which have one tensors as input and change the shape and stride of it. 
    '''
    @final
    @classmethod
    def forward(cls, a : Tensor, _retain_grad : bool = False) -> Tensor:
        k = cls.forward_kernel(a.shape, a.stride)
        return Tensor(data = None,
                      shape = None,
                      requires_grad = a._requires_grad,
                      dtype = None,
                      comes_from  = Expression([a], cls.backward),
                      _retain_grad = _retain_grad,
                      _k = k)   
        
    @final
    @classmethod
    def backward(cls, t : tuple[Tensor], seed : KernelTensor) -> None:
        derive(t[0], seed, cls.backward_kernel(t[0].shape, t[0].stride))
        
    
    @staticmethod
    def forward_kernel(t : KernelTensor, shape : tuple[int], stride : tuple[int]) -> KernelTensor:
        raise NotImplementedError("Subclasses must implement 'forward_kernel' function")
   
    @staticmethod
    def backward_kernel(t : KernelTensor,
                        input_shape : tuple[int], 
                        input_stride : tuple[int],
                        seed : KernelTensor) -> KernelTensor:
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

#TODO Struct this in C?
@dataclass(frozen=True)
class KernelTensor:
    array : tuple[float] #TODO dtypes
    memory_length : int
    shape : tuple[int]
    dtype : type


@dataclass(frozen=True)
class Expression:
    inputs: tuple[Tensor]
    backward: Callable


def derive(a : Tensor, seed : KernelTensor) -> None:
    if a.grad is not None:
        #TODO should there be kernel add here bc seed stride may not match partial array stride
        a.grad += seed

    if a.comes_from is not None:
         a.comes_from.backward(a.comes_from.inputs, seed)



class Tensor(): 
    """
    If data is None then one or both of shape and _array will be used.
    If only shape is passed, a tensor of that shape will be created with uninitialized values.
    If only _array is passed, a 1d tensor will be created with shape [len(_array)].
    If shape and stride are passed, the stride will be validated.
    """

    def __init__(self, 
                 array : tuple[float],
                 shape : tuple[int],
                 requires_grad : bool = False, 
                 dtype : type  = float, 
                 comes_from : Expression = None,
                 _retain_grad : bool = False,
                 _k : KernelTensor = None,
                ):
        
        self.comes_from = comes_from
        
        self._requires_grad = requires_grad

        if _retain_grad is None: #This means tensor created by user
                self._retain_grad = self._requires_grad 
        else: #This means the tensor is created by a function 
            self._retain_grad = _retain_grad 
        
        if self._retain_grad:
                assert self._requires_grad == True, "_retain_grad can only be True is requires_grad is True"

        if _k is None:           
            if self._retain_grad:
                self.grad = KernelTensor(array=[0.]*len(array), 
                                                    memory_length=len(array), 
                                                    shape=shape, 
                                                    dtype=dtype)
            else:
                self.grad = None

            self._k = KernelTensor(array=array, 
                                    memory_length=len(array), 
                                    shape=shape, 
                                    dtype=dtype)
        else:
            self._k = _k

            if self._retain_grad:
                self.grad = KernelTensor(array=[0.]*_k.memory_length, 
                                                    memory_length=_k.memory_length, 
                                                    shape=_k.shape, 
                                                    dtype=_k.dtype)
            else:
                self.grad = None
        
        self.stride = get_contiguous_stride(shape)
        
        assert self._numel(self._k.shape) == self._k.memory_length, "Dimensions of shape must match same number of elements of array"
        assert self.is_stride_shape_compatible(self._k.shape, self.stride), "Stride not compatible with shape"
        
        #TODO remove this when times comes
        assert type(self._k.array) == tuple, "Array must be immutable"
        assert type(self._k.shape) == tuple, "Shape must be immutable"


    @property
    def array(self) -> tuple[float]:
        return self._k.array

    @property
    def shape(self) -> tuple[int]:
        return self._k.shape
    
    @property
    def dtype(self) -> type:
        return self._k.dtype

    #TODO unit tests
    @classmethod
    def is_stride_shape_compatible(self, shape, stride):
        if len(shape) != len(stride):
            return False
        
        if any(s < 0 for s in stride):
            return False
        
        if any(size > 0 and stride == 0 for size, stride in zip(shape, stride)):
            return False
        
        seen = set()
        def check_overlap(dim, offset):
            if dim == len(shape):
                if offset in seen:
                    return False
                seen.add(offset)
                return True
                
            for i in range(shape[dim]):
                if not check_overlap(dim + 1, offset + i * stride[dim]):
                    return False
            return True
        return check_overlap(0, 0)
    
    
    #TODO need this?
    # @classmethod
    # def _get_shape(self,lst):
    #     def _get_size( lst):
    #         if isinstance(lst, list):
    #             return [len(lst)] + _get_size(lst[0]) if len(lst) > 0 else [0]
    #         else:
    #             return []
    #     return tuple(_get_size( lst))

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
            
        string = format_array(self.array, self.shape, self.stride) + \
                f"\nShape: {self.shape}\nStride: {self.stride}\nDtype: {self.dtype}"
        
        if self._requires_grad:
            string += f"\nBackward: "
            if self.comes_from is not None:
                string += f"{self.comes_from.backward}"
            else:
                string += "None"
                
            if self._retain_grad:
                string += "\n" + format_array(self.grad.array, 
                                            self.shape, 
                                            self.stride)
                
        return string

    def retain_grad(self, set_ret_grad : bool) -> None:
        
        #TODO _retain_grad or retain_grad?
        if set_ret_grad == True and self._retain_grad == True:
            return #we do not want to overwrite grad
        
        self._retain_grad = set_ret_grad 
        if set_ret_grad:
            self.requires_grad(True)
            #TODO partial array types
            self.grad = KernelTensor(array=[0.]*len(self._k.memory_length), 
                                               memory_length=self._k.memory_length, 
                                               shape=self.shape, 
                                               dtype=self.dtype)
        else:
            self.grad = None
        
    def requires_grad(self, set_req_grad : bool) -> None:
        self._requires_grad = set_req_grad
        if self._requires_grad == False:
            self._retain_grad = False
            self.grad = None 
        
    @classmethod
    def _numel(self, shape : tuple[int]) -> int:
        result = 1  
        for num in shape:
            result *= num
        return result
 

    @property
    def numel(self) -> int:
        self._numel(self.shape)

    
    @property
    def dim(self) -> int:
        return len(self.shape)
    
    def backward(self) -> None:
        #TODO need both asserts? edge cases? THINK
        assert self.numel == 1, "Backward is only defined when Tensor is a scalar"
        assert self._k.memory_length == 1, "Backward is only defined when Tensor is a scalar"
        seed = KernelTensor(array=[0.], 
                            memory_length=1, 
                            shape=[1], 
                            dtype=self.dtype)
        derive(self, seed)

def get_contiguous_stride(shape : tuple[int]) -> tuple[int]:
        stride = [1]
        for s in shape[-1:0:-1]:  
            stride.append(stride[-1] * s)
        return tuple(reversed(stride))

    
    


    





