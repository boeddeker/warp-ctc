
# distutils: language = c++
# distutils: libraries = warpctc

import numpy as np
cimport numpy as np
import numpy
cimport numpy

from cpython cimport array
import array
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free

cdef extern from "driver_types.h":
    enum cudaMemcpyKind:
        cudaMemcpyHostToHost
        cudaMemcpyHostToDevice
        cudaMemcpyDeviceToHost
        cudaMemcpyDeviceToDevice
        cudaMemcpyDefault

    enum cudaError:
        cudaSuccess

    ctypedef cudaError cudaError_t

    ctypedef struct CUstream_st:
        pass
    ctypedef CUstream_st *cudaStream_t

    cudaError_t cudaStreamCreate	(	cudaStream_t * 	pStream	 )



cdef extern from "cuda_runtime_api.h":
    cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
                           cudaMemcpyKind kind)
    cudaError_t cudaMalloc(void **devPtr, size_t size)
    cudaError_t cudaFree(void *devPtr)
    const char* cudaGetErrorString(cudaError_t error)

    cudaError_t cudaDeviceSynchronize()
    cudaError_t cudaGetLastError()

    cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                                cudaMemcpyKind kind, 		cudaStream_t 	stream  )
    cudaError_t cudaSetDevice(int device)

#cdef extern from "/net/home/boeddeker/python/github/warp-ctc/include/ctc.h":
cdef extern from "ctc.h":
    enum ctcStatus_t:
        CTC_STATUS_SUCCESS = 0,
        CTC_STATUS_MEMOPS_FAILED = 1,
        CTC_STATUS_INVALID_VALUE = 2,
        CTC_STATUS_EXECUTION_FAILED = 3,
        CTC_STATUS_UNKNOWN_ERROR = 4
    enum ctcComputeLocation:
        CTC_CPU = 0,
        CTC_GPU = 1
            
    struct ctcComputeInfo:
        ctcComputeLocation loc
        unsigned int num_threads
        
    ctcStatus_t compute_ctc_loss(const float* const activations,
                             float* gradients,
                             const int* const flat_labels,
                             const int* const label_lengths,
                             const int* const input_lengths,
                             int alphabet_size,
                             int minibatch,
                             float *costs,
                             void *workspace,
                             ctcComputeInfo info)
    ctcStatus_t get_workspace_size(const int* const label_lengths,
                               const int* const input_lengths,
                               int alphabet_size, int minibatch,
                               ctcComputeInfo info,
                               size_t* size_bytes)
   

def ctc(activations, labels, use_gpu = False, num_threads = 1):
    """
    activations.shape == (T, minibatch, alphabet_size)
    labels.shape == (L,)

    GPU is currently not working
    """

    print('Startasdof dsf')

    T, minibatch, alphabet_size = activations.shape
    
    # activations = numpy.ascontiguousarray(activations).ravel()
    activations = activations.ravel()

    label_lengths = numpy.shape(labels)[1:] # = (L,)
    labels = labels.ravel()

    lengths = np.array([T])

    cdef ctcComputeInfo info;
    if use_gpu:
        info.loc = CTC_GPU
    else:
        info.loc = CTC_CPU

    info.num_threads = num_threads
    cdef size_t cpu_alloc_bytes

    cdef vector[int] labels_vec = labels;
    # cdef vector[int] labels_vec = [1, 2];
    cdef vector[int] label_lengths_vec = label_lengths;
    # cdef vector[int] label_lengths_vec = [2];
    cdef vector[int] lengths_vec = lengths;
    # cdef vector[int] lengths_vec
    # lengths_vec.push_back(T)

    print('activations type: ', type(activations))
    print('activations type: ', type(activations.data))

    cdef vector[float] activations_vec = activations;
    cdef float* activations_array

    if use_gpu == False:
        activations_array = <float*><void*>activations.data
    else:
        activations_array = <float*><void*>activations.data.ptr

    cdef float score = 0

    print('activations_vec: ', type(activations_vec))


    cdef int status = get_workspace_size(label_lengths_vec.data(), lengths_vec.data(),
                                          alphabet_size, lengths.shape[0], info,
                                          &cpu_alloc_bytes)

    # cdef cudaStream_t stream
    # cudaStreamCreate(&stream)

    # cdef float *activations_gpu;
    # cudaMalloc(<void **>&activations_gpu,
    #                activations.size * sizeof(float))
    # cudaMemcpyAsync(activations_gpu, activations_array,
    #                                activations.size * sizeof(float),
    #                                cudaMemcpyHostToDevice, stream)

    cdef void* ctc_workspace;
    cdef vector[float] grads;

    cdef float* grads_pointer = grads.data()
    cdef void* grads_pointer_void = grads_pointer

    if use_gpu == False:
        grads = numpy.empty((T * minibatch * alphabet_size), dtype=float, order='C')
        grads_pointer = grads.data()
        ctc_workspace = malloc(cpu_alloc_bytes)
        if not ctc_workspace:
            raise MemoryError()
    else:
        if cudaMalloc(&grads_pointer_void, (alphabet_size * T) * sizeof(float)):
            raise MemoryError()
        if cudaMalloc(&ctc_workspace, cpu_alloc_bytes):
            raise MemoryError()

    try:
        # grads_pointer
        if use_gpu == False:
            status = compute_ctc_loss(activations_vec.data(), grads_pointer,
                                            labels_vec.data(), label_lengths_vec.data(),
                                            lengths_vec.data(),
                                            alphabet_size,
                                            lengths_vec.size(),
                                            &score,
                                            ctc_workspace,
                                            info)
        else:
            print('Start gpu')
            #print('activations_gpu: ', [activations_gpu[i] for i in range(activations.size)])
            status = compute_ctc_loss(activations_array, NULL,
                                            labels_vec.data(), label_lengths_vec.data(),
                                            lengths_vec.data(),
                                            alphabet_size,
                                            lengths_vec.size(),
                                            &score,
                                            ctc_workspace,
                                            info)
            print('End gpu')

        if status != CTC_STATUS_SUCCESS:
            print('ERROR: status: ', status)
            if status == CTC_STATUS_MEMOPS_FAILED:
                raise MemoryError()
            elif status == CTC_STATUS_INVALID_VALUE:
                raise ValueError()
            elif status == CTC_STATUS_EXECUTION_FAILED:
                raise RuntimeError()
            elif status == CTC_STATUS_UNKNOWN_ERROR:
                raise RuntimeError()
            else:
                raise RuntimeError()
    finally:
        # return the previously allocated memory to the system
        if use_gpu == False:
            free(ctc_workspace)
        else:
            cudaFree(ctc_workspace)
            # cudaFree(activations_gpu)

    print('Score: ', score)
    
    return numpy.array(grads).reshape((T, minibatch, alphabet_size)), score


