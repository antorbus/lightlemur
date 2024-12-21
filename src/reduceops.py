from metaops import ReduceOperation
from tensor import KernelTensor

class mean(ReduceOperation):    
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




class sum(ReduceOperation):    
    @staticmethod
    def forward_kernel(t : KernelTensor, dim : int) -> KernelTensor:
        assert dim == 5, "sum not implemented for other dimensions yet"
        reduced_shape = (t.shape[0],t.shape[1],t.shape[2],t.shape[3],1)
        reduced_memory_length = t.shape[0]*t.shape[1]*t.shape[2]*t.shape[3]
        arr = [None]*reduced_memory_length
        for r_idx in range(reduced_memory_length):
            dim_total = 0.
            for i in range(t.shape[4]):
                dim_total += t.array[r_idx*t.shape[4] + i]
            arr[r_idx] = dim_total

        return KernelTensor(tuple(arr), reduced_memory_length, reduced_shape, t.dtype)
   
    @staticmethod
    def backward_kernel(t: KernelTensor, dim: int, seed : KernelTensor) -> KernelTensor:
        pass
        #seed must be 1d TODO
        # n_elements = t.numel
        # next_seed = np.ones((t.numel,), dtype = t.dtype) * (seed/n_elements)
        # return next_seed
