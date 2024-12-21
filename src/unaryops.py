
from metaops import UnaryOperation
from tensor import KernelTensor

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

    

