#pragma once

#include <driver_types.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>

#include "utils.hpp"

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
  protected:
    double *data_device_, *c_device_, *old_c_device_;
    size_t *labels_;
    const size_t d_sz_, c_sz_, dim_;
  public:
    KmeansStrategyGpuGlobalBase(const size_t sz, const size_t k, const size_t dim)
      : d_sz_{sz}, c_sz_{k}, dim_{dim}
    {
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
    bool converged(double *host_c, double *host_old_c) override;
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
    using KmeansStrategyGpuGlobalBase::data_device_;
    using KmeansStrategyGpuGlobalBase::c_device_;
    using KmeansStrategyGpuGlobalBase::old_c_device_;
    using KmeansStrategyGpuGlobalBase::labels_;
    using KmeansStrategyGpuGlobalBase::d_sz_;
    using KmeansStrategyGpuGlobalBase::c_sz_;
    using KmeansStrategyGpuGlobalBase::dim_;

  public:
    void findNearestCentroids() override ;
    void averageLabeledCentroids() override;
    ~KmeansStrategyGpuBaseline() override {}
  };

  class KmeansGpu : public KmeansBase<KmeansGpu>
  {
    KmeansStrategy * stgy_;
  public:
    KmeansGpu(const Data &d, const bool random, const size_t n_clu, const unsigned max_iters)
      : KmeansBase<KmeansGpu>(d, random, n_clu, max_iters)
    {}
    Labels fit() override ;
    Centroids<double> & result() override { 
      return KmeansBase<KmeansGpu>::solved_ 
              ? KmeansBase<KmeansGpu>::c_ 
              : (fit(), KmeansBase<KmeansGpu>::c_); 
    }
  };
}