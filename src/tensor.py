from __future__ import annotations
from dataclasses import dataclass
from typing import Callable


#TODO Struct this in C?
class KernelTensor:
    def __init__(self, array : tuple[float], 
                 memory_length : int, 
                 shape : tuple[int], 
                 dtype : type): 
        # TODO: refine dtypes
        
        self.__dict__['_frozen'] = False

        self.array = array
        self.memory_length = memory_length
        self.shape = shape
        self.dtype = dtype
        self.stride = self.get_contiguous_stride(shape)

        if len(self.shape) != 5:
            raise ValueError(f"The shape must be a tuple of length 5, got {len(self.shape)}")
        if len(self.stride) != 5:
            raise ValueError(f"The stride must be a tuple of length 5, got {len(self.stride)}")
              
        assert self._numel(self.shape) == self.memory_length, "Dimensions of shape must match same number of elements of array"
        assert self.is_stride_shape_compatible(self.shape, self.stride), "Stride not compatible with shape"
        
        #TODO Can remove? add check for dtype
        assert type(self.array) == tuple, "Array must be immutable"
        assert type(self.memory_length) == int, "Memory length must be an int"
        assert type(self.shape) == tuple, "Shape must be immutable"
        assert type(self.stride) == tuple, "Stride must be immutable"


        self.__dict__['_frozen'] = True

    @staticmethod
    def get_contiguous_stride(shape : tuple[int]) -> tuple[int]:
            stride = [1]
            for s in shape[-1:0:-1]:  
                stride.append(stride[-1] * s)
            return tuple(reversed(stride))
    
    @staticmethod
    def _numel(shape : tuple[int]) -> int:
        result = 1  
        for num in shape:
            result *= num
        return result
    
    #TODO unit tests
    @staticmethod
    def is_stride_shape_compatible(shape, stride):
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

    def __setattr__(self, key, value):
        if self.__dict__.get('_frozen', False):
            raise AttributeError(f"Cannot modify attribute '{key}' in a frozen instance")
        super().__setattr__(key, value)

    def __delattr__(self, item):
        if self.__dict__.get('_frozen', False):
            raise AttributeError(f"Cannot delete attribute '{item}' in a frozen instance")
        super().__delattr__(item)


@dataclass(frozen=True)
class Expression:
    inputs: tuple[Tensor]
    backward: Callable


def derive(a : Tensor, seed : KernelTensor) -> None:
    if a.grad is not None:
        #TODO should there be kernel add here bc seed stride may not match partial array stride
        a.grad += seed #This messes us up if Kernel tensor array is a tuple

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
                 shape : tuple[int] = None,
                 requires_grad : bool = False, 
                 dtype : type  = float, 
                 comes_from : Expression = None,
                 _retain_grad : bool = None, #must be None
                 _k : KernelTensor = None,
                ):
        if type(array) == list:
            array = tuple(array)
        if shape is None:
            shape = (len(array),)
        if len(shape)<5:
            shape = (1,) * (5 - len(shape)) + shape
        
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
                self.grad = KernelTensor(array=tuple([0.]*len(array)), 
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
                self.grad = KernelTensor(array=tuple([0.]*_k.memory_length), 
                                                    memory_length=_k.memory_length, 
                                                    shape=_k.shape, 
                                                    dtype=_k.dtype)
            else:
                self.grad = None
        
  
    @property
    def stride(self) -> tuple[float]:
        return self._k.stride

    @property
    def array(self) -> tuple[float]:
        return self._k.array

    @property
    def shape(self) -> tuple[int]:
        return self._k.shape
    
    @property
    def dtype(self) -> type:
        return self._k.dtype
        
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
            self.grad = KernelTensor(array=tuple([0.]*len(self._k.memory_length)), 
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
        

    @property
    def numel(self) -> int:
        return self._k._numel(self.shape)

    
    @property
    def dim(self) -> int:
        return len(self.shape)
    
    def backward(self) -> None:
        #TODO need both asserts? edge cases? THINK
        assert self.numel == 1, "Backward is only defined when Tensor is a scalar"
        assert self._k.memory_length == 1, "Backward is only defined when Tensor is a scalar"
        seed = KernelTensor(array=(0.,), 
                            memory_length=1, 
                            shape=(1,1,1,1,1), 
                            dtype=self.dtype)
        derive(self, seed)

    
    


    


