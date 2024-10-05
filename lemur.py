import numpy as np
from __future__ import annotations
from typing import final
from dataclasses import dataclass


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
        
    
    @final
    @classmethod
    def forward(cls, a: Tensor, _retain_grad : bool = False) -> Tensor:
        return Tensor(data = None,
                      _array = cls.forward_kernel(a),
                      shape = a.shape,
                      requires_grad = a._requires_grad,
                      dtype = a.dtype,
                      comes_from  = Expression([a],cls.backward),
                      _retain_grad = _retain_grad)        
        
    @final
    @classmethod
    def backward(cls, t : list[Tensor], seed : np.ndarray) -> None:
        #Since unary, there will only be one element in t 
        derive(t[0], cls.backward_kernel(t[0], seed))
        
    @staticmethod
    def forward_kernel(a : Tensor) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement 'forward_kernel' function")
   
    @staticmethod
    def backward_kernel(a: Tensor, seed : np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement 'backward_kernel' function")

class BinaryOperation(metaclass=OperationMeta):
    '''
    Functions which have two tensors as input and one as output, sizes can differ. 
    '''
    @final
    @classmethod
    def forward(cls, a: Tensor, b: Tensor, _retain_grad : bool = False) -> Tensor:
        _array, shape = cls.forward_kernel(a, b)
        return Tensor(data = None,
                      _array = _array,
                      shape = shape,
                      requires_grad = a._requires_grad or b._requires_grad,
                      dtype = a.dtype,
                      comes_from  = Expression([a,b],cls.backward),
                      _retain_grad = _retain_grad)        
        
    @final
    @classmethod
    def backward(cls, t : list[Tensor], seed : np.ndarray) -> None:
        #TODO: This can probably be multithread 
        #for idx, tensor in enumerate(t):
        derive(t[0], cls.backward_kernel(t, seed, 0))
        derive(t[1], cls.backward_kernel(t, seed, 1))
        
    @staticmethod
    def forward_kernel(a : Tensor, b : Tensor) -> (np.ndarray, list[int]):
        raise NotImplementedError("Subclasses must implement 'forward_kernel' function")
   
    @staticmethod
    def backward_kernel(t: list[Tensor], seed : np.ndarray, idx : int) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement 'backward_kernel' function")

class ReduceOperation(metaclass=OperationMeta):
    '''
    Functions which have one tensors as input and one as output, sizes will differ. 
    '''
    ##TODO reduce along ONE DIMENSION
    @final
    @classmethod
    def forward(cls, a : Tensor, _retain_grad : bool = False) -> Tensor:
        _array, shape = cls.forward_kernel(a)
        return Tensor(data = None,
                      _array = _array,
                      shape = shape,
                      requires_grad = a._requires_grad,
                      dtype = a.dtype,
                      comes_from  = Expression([a],cls.backward),
                      _retain_grad = _retain_grad)        
        
    @final
    @classmethod
    def backward(cls, t : list[Tensor], seed : np.ndarray) -> None:
        derive(t[0], cls.backward_kernel(t[0], seed))
        
    @staticmethod
    def forward_kernel(a : Tensor,) -> (np.ndarray, list[int]):
        raise NotImplementedError("Subclasses must implement 'forward_kernel' function")
   
    @staticmethod
    def backward_kernel(t: Tensor, seed : np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement 'backward_kernel' function")

               
        
@dataclass(frozen=True)
class Expression:
    inputs: list[Tensor]
    backward: Backward
        

class Tensor(): 
    """
    Currently data cannot be passed alongside _array and shape. 
    If data is None then one or both of shape and _array will be used.
    If both shape and _array are present the array will be viewed as the shaped passed.
    If only shape is passed, a tensor of that shape will be created with uninitialized values.
    If only _array is passed, a 1d tensor will be created with shape [len(_array)].
    """
    def __init__(self, 
                 data : list[list[...]], 
                 _array : np.ndarray = None,
                 shape : list[int] = None,
                 requires_grad : bool = False, 
                 dtype : type  = float, 
                 comes_from : Expression = None,
                 _retain_grad : bool = None,
                ):
        
        
        if data is not None:
            assert _array is None, "Either pass in data as a list, or an _array."
            assert shape is None, "Changing the shape cannot be passed as a parameter if data is passed."
            self._array  = np.array(data, dtype=dtype).flatten()
            self.shape = self._get_size(data)
              
        elif shape is not None: 
            if _array is not None:
                self._array = _array.flatten()
                self.shape = shape
            else:
                self._array  = np.empty(shape, dtype=dtype).flatten()
                self.shape = shape
        
        elif _array is not None:
            self._array = _array.flatten()
            self.shape = [len(self._array)]
             
        else:
            raise ValueError("Cannot create tensor without data, shape, or _array.")
        
        assert self.numel == len(self._array), ValueError("Dimensions must be regular")
        
        self.dtype = dtype
        
        self._requires_grad = requires_grad
        
        if _retain_grad is None: #This means tensor created by user
            self._retain_grad = self._requires_grad 
        else:
            self._retain_grad = _retain_grad #This means the tensor is created by a function 
        
        if self._retain_grad:
            assert self._requires_grad == True, "_retain_grad can only be True is requires_grad is True"
            self._partial_array = np.full(self.numel, 0, dtype=dtype)
        else:
            self._partial_array = None
        
        self.comes_from = comes_from
        
    def __repr__(self):
        def format_array(array, shape):
            if len(shape) == 0:  
                return '[]'
            elif len(shape) == 1: 
                return str(array)
            else:
                result = []
                stride = 1 if shape[0] == 0 else len(array) // shape[0]  # Handle empty array case
                for i in range(shape[0]):
                    sub_array = array[i * stride:(i + 1) * stride]
                    result.append(format_array(sub_array, shape[1:]))
                return "[" + ",\n ".join(result) + "]"
            
        string = format_array(self._array, self.shape) + f"\nShape: {self.shape}\nDtype: {self.dtype}" 
        if self._requires_grad:
            string += f"\nBackward: "
            if self.comes_from is not None:
                string += f"{self.comes_from.backward}"
            else:
                string += "None"
                
            if self._retain_grad:
                string +=  "\n"+format_array(self._partial_array, self.shape)
                
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
    
    @classmethod
    def _get_size(self, lst):
        if isinstance(lst, list):
            return [len(lst)] + self._get_size(lst[0]) if len(lst) > 0 else [0]
        else:
            return []
    
    @property
    def dim(self):
        return len(self.shape)
    
    def backward(self):
        assert self.numel == 1, "Backward is only defined when Tensor is a scalar"
        seed = np.array([1],dtype = self.dtype)
        derive(self, seed)

def derive(a : Tensor, seed : np.ndarray ):
    if a._partial_array is not None:
        a._partial_array += seed
    if a.comes_from is not None:
         a.comes_from.backward(a.comes_from.inputs, seed)
    
    

        

class relu(UnaryOperation):
    @staticmethod
    def forward_kernel(a : Tensor) -> np.ndarray:
        return a._array * (a._array > 0)  

    @staticmethod
    def backward_kernel(a: Tensor, seed : np.ndarray) -> np.ndarray: #returns next seed        
        next_seed = seed
        next_seed[a._array < 0]  = 0
        return next_seed

    
class addTensors(BinaryOperation):
    @staticmethod
    def forward_kernel(a : Tensor, b : Tensor) -> (np.ndarray, list[int]):
        assert a.shape == b.shape, "Cannot add two tensors of different size"
        return a._array + b._array, a.shape
   
    @staticmethod
    def backward_kernel(t: list[Tensor], seed : np.ndarray, idx : int) -> np.ndarray:
        return seed

class hadamardProduct(BinaryOperation):
    @staticmethod
    def forward_kernel(a : Tensor, b : Tensor) -> (np.ndarray, list[int]):
        assert a.shape == b.shape, "Cannot do hadamard product of tensors of different size"
        return a._array * b._array, a.shape
   
    @staticmethod
    def backward_kernel(t: list[Tensor], seed : np.ndarray, idx : int) -> np.ndarray:
        if idx == 0:
            return seed * t[1]._array
        if idx == 1:
            return seed * t[0]._array
      


class matmul(BinaryOperation):
    '''Can matmul tensots of same number dimensions or, matrix of 2 dimensions and tensor of more.'''
    @staticmethod
    def forward_kernel(a : Tensor, b : Tensor) -> (np.ndarray, list[int]):
        if not (a.dim == 2 or b.dim == 2):
            assert a.dim == b.dim, "Cannot perform matmul with tensors of different dimensions"
        
        assert a.shape[-1] == b.shape[-2], f"Size mismatch, cannot multiply tensors of sizes {a.shape} {b.shape}"
        #TODO this does not work, must get b shape too
        new_shape = a.shape.copy()
        new_shape[-1] = b.shape[-1]
        
        new_array = a._array.reshape(a.shape) @ b._array.reshape(b.shape)
        return new_array.flatten(), new_shape
   
    @staticmethod
    def backward_kernel(t: list[Tensor], seed : np.ndarray, idx : int) -> np.ndarray:
        
        seed_shape = t[0].shape.copy()
        seed_shape[-1] = t[1].shape[-1]
 
   
        if idx == 0:
            T = t[1]._array.reshape(t[1].shape)
            T = np.moveaxis(T, -1, -2)
            new_seed = (seed.reshape(seed_shape) @ T)
        if idx == 1:
            T = t[0]._array.reshape(t[0].shape)
            T = np.moveaxis(T, -1, -2)
            new_seed = (T @ seed.reshape(seed_shape)) 
        if t[idx].dim == 2:
            sum_over = tuple(range(len(seed_shape) - 2))
            new_seed = new_seed.sum(axis = sum_over)
        return new_seed.flatten()


class mean(ReduceOperation):    
    @staticmethod
    def forward_kernel(t : Tensor) -> (np.ndarray, list[int]):
        return t._array.mean(), [1]
   
    @staticmethod
    def backward_kernel(t: Tensor, seed : np.ndarray) -> np.ndarray:
        #seed must be 1d TODO
        n_elements = t.numel
        next_seed = np.ones((t.numel,) , dtype = t.dtype) * (seed/n_elements)
        return next_seed
     

class sigmoid(UnaryOperation):
    @staticmethod
    def forward_kernel(a : Tensor) -> np.ndarray:
        return 1/(1 + np.exp(-a._array))

    @staticmethod
    def backward_kernel(a: Tensor, seed : np.ndarray) -> np.ndarray: 
        sigmoid_input = sigmoid.forward_kernel(a)
        return seed * sigmoid_input * (1 - sigmoid_input)
    