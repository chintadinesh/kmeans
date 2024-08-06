#pragma once

#include <driver_types.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>
#include <map>
#include <utility>

#include "utils.hpp"
#include "args.hpp"

extern utils::DebugStream dbg;

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
  public:
    enum EventType {MEMCPY = 0, CLASSIFY, UPDATE, OTHERS, _EVENT_TYPE_LEN};
  protected:
    double *data_device_, *c_device_, *old_c_device_;
    unsigned *labels_device_;
    const size_t d_sz_, c_sz_, dim_;

    std::map<EventType, std::vector<float>> event_times_;

    cudaEvent_t start, stop;

  protected:
    inline void registerTime(EventType ev, float time){
      auto it = event_times_.find(ev);
      if(it == event_times_.end()){
        event_times_.insert(std::make_pair(ev, std::vector{time}));
      }
      else{
        it->second.push_back(time);
      }
    }

    inline void startGpuTimer() { cudaEventRecord(start, 0); }
    inline float endGpuTimer() { 
      cudaEventRecord(stop, 0); 
      cudaEventSynchronize(stop);
      float elapsed_time;
      cudaEventElapsedTime(&elapsed_time, start, stop);
      return elapsed_time;
    }

  public:

    KmeansStrategyGpuGlobalBase(const size_t sz, const size_t k, const size_t dim);
    void init(const double *d, const double *c, const size_t data_sz, const size_t c_sz) override; 
    void collect(double *c, unsigned *l, const size_t c_sz, const size_t l_sz) override; 
    void getCentroids(double *host_c) override;
    void swap() override; 
    ~KmeansStrategyGpuGlobalBase() override; 
  };

  class KmeansStrategyGpuBaseline : public KmeansStrategyGpuGlobalBase 
  {
    using KmeansStrategyGpuGlobalBase::data_device_;
    using KmeansStrategyGpuGlobalBase::c_device_;
    using KmeansStrategyGpuGlobalBase::old_c_device_;
    using KmeansStrategyGpuGlobalBase::labels_device_;
    using KmeansStrategyGpuGlobalBase::d_sz_;
    using KmeansStrategyGpuGlobalBase::c_sz_;
    using KmeansStrategyGpuGlobalBase::dim_;

  public:
    KmeansStrategyGpuBaseline(const size_t sz, const size_t k, const size_t dim)
      : KmeansStrategyGpuGlobalBase(sz, k, dim)
    {}

    void findNearestCentroids() override ;
    void averageLabeledCentroids() override;
    ~KmeansStrategyGpuBaseline() override {}
  };

  class KmeansGpu : public KmeansBase<KmeansGpu>
  {
    std::unique_ptr<KmeansStrategy> stgy_;
  public:
    KmeansGpu(const Data &d, 
              const bool random, 
              const size_t n_clu, 
              const unsigned max_iters,
              KmeansStrategy * stgy)
      : KmeansBase<KmeansGpu>(d, random, n_clu, max_iters), stgy_{stgy}
    {}
    Labels fit() override ;
    Centroids<double> & result() override { 
      return KmeansBase<KmeansGpu>::solved_ 
              ? KmeansBase<KmeansGpu>::c_ 
              : (fit(), KmeansBase<KmeansGpu>::c_); 
    }
  };
}
