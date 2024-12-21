from tensor import Tensor, Expression, KernelTensor, derive
from typing import final

class OperationMeta(type):
    def __new__(cls, name, bases, dct):
        def has_required_methods(class_dict):
            required_methods = ['__call__', 'backward', 'forward_kernel', 'backward_kernel']
            return all(any(callable(getattr(base, meth, None)) or meth in class_dict for base in bases + (class_obj,))
                       for meth in required_methods)
        
        class_obj = super().__new__(cls, name, bases, dct)
        
        if not has_required_methods(dct):
            missing_methods = [
                meth for meth in ['__call__', 'backward', 'forward_kernel', 'backward_kernel']
                if not any(callable(getattr(base, meth, None)) or meth in dct for base in bases + (class_obj,))
            ]
            raise AttributeError(f"Class {name} is missing required methods: {', '.join(missing_methods)}")
        
        return class_obj

    
class UnaryOperation(metaclass=OperationMeta):
    '''
    Element-wise functions which have one tensor as input and one of the same shape as output. 
    '''
    
    
    #TODO WHEN NOT USING NUMPY NEED TO MAKE A COPY OF data
    
    @final
    @classmethod
    def __call__(cls, a: Tensor, _retain_grad : bool = False) -> Tensor:

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


class BinaryOperation(metaclass=OperationMeta):
    '''
    Functions which have two tensors of the same shape as input and one as 
    output of the same shape. 
    '''
    @final
    @classmethod
    def __call__(cls, a: Tensor, b: Tensor, _retain_grad : bool = False) -> Tensor:

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


class ReduceOperation(metaclass=OperationMeta):
    '''
    Functions which have one tensors and a dimension as input, the output
    is a tensor with one dimension less. 
    '''
    @final
    @classmethod
    def __call__(cls, a : Tensor, dim : int, _retain_grad : bool = False) -> Tensor:

        assert dim < a.dim, f"Cannot reduce along dimension {dim} when tensor.dim is {a.dim}"

        k = cls.forward_kernel(a._k, dim)
        assert k.dtype == a.dtype, "Reduce operation must perserve dtype."
        assert k.shape[dim] == 1, "Reduce did not remove one dimension."

        #TODO check memory length reduce appropiately

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
    


class ShapeOperation(metaclass=OperationMeta):
    '''
    Functions which have one tensors as input and change the shape and stride of it. 
    '''
    @final
    @classmethod
    def __call__(cls, a : Tensor, _retain_grad : bool = False) -> Tensor:
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
        

