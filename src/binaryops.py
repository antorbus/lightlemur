from metaops import BinaryOperation
from tensor import KernelTensor


class add(BinaryOperation):
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

class hadamard_product(BinaryOperation):
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
      

