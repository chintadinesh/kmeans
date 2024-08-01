#pragma once

#include "utils.hpp"
#include <driver_types.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

namespace utils {
  // all the memory is global memory
  class KmeansStrategyGpuGlobalBase : public KmeansStrategy 
  {
    void *data_device_, *c_device_, *old_c_device_;
    unsigned *labels_;
  public:
    KmeansStrategyGpuGlobalBase(const size_t dim, const size_t sz, const size_t k){
      gpuErrchk( cudaMalloc((void**)&data_device_, sizeof(double)*dim*sz) );
      gpuErrchk( cudaMalloc((void**)&c_device_, sizeof(double)*dim*k) );
      gpuErrchk( cudaMalloc((void**)&old_c_device_, sizeof(double)*dim*k) );
      gpuErrchk( cudaMalloc((void**)&labels_, sizeof(unsigned)*sz) );
    }
    void init(const double *d, const double *c, const size_t data_sz, const size_t c_sz) override {
      gpuErrchk( cudaMemcpy(data_device_, d, data_sz, cudaMemcpyHostToDevice) );
      gpuErrchk( cudaMemcpy(c_device_, c, c_sz, cudaMemcpyHostToDevice) );
      gpuErrchk( cudaMemcpy(old_c_device_, c, c_sz, cudaMemcpyHostToDevice) );
    }
    void collect(double *c, unsigned *l, const size_t c_sz, const size_t l_sz) override {
      gpuErrchk( cudaMemcpy(c, c_device_, c_sz, cudaMemcpyDeviceToHost) );
      gpuErrchk( cudaMemcpy(l, labels_, l_sz, cudaMemcpyDeviceToHost) );
    }
    void swap() override { std::swap(c_device_, old_c_device_); }
    ~KmeansStrategyGpuGlobalBase() override {
      gpuErrchk( cudaFree(data_device_) );
      gpuErrchk( cudaFree(c_device_) );
      gpuErrchk( cudaFree(old_c_device_) );
      gpuErrchk( cudaFree(labels_) );
    }
  };

  class KmeansStrategyGpuBaseline : public KmeansStrategyGpuGlobalBase 
  {
    void findNearestCentroids() override ;
    void averageLabeledCentroids() override;
    bool converged() override;
  };

  class KmeansStrategyGpuSharedBase : public KmeansStrategy 
  {
  };

  template<typename ElemType>
  class KmeansGpu : public KmeansBase<KmeansGpu<ElemType>, ElemType>
  {
    KmeansStrategy * stgy_;
  public:
    KmeansGpu(const Data &d, const bool random, const size_t n_clu, const unsigned max_iters)
      : KmeansBase<KmeansGpu<ElemType>, ElemType>(d, random, n_clu, max_iters)
    {}
    Labels fit() override ;
    Centroids<ElemType> & result() override { 
      return KmeansBase<KmeansGpu<ElemType>, ElemType>::solved_ 
              ? KmeansBase<KmeansGpu<ElemType>, ElemType>::c_ 
              : (fit(), KmeansBase<KmeansGpu<ElemType>, ElemType>::c_); 
    }
  };
}