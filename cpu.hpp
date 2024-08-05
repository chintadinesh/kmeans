#pragma once
#include "utils.hpp"

extern utils::DebugStream dbg;

namespace utils {
  class KmeansCpu : public KmeansBase<KmeansCpu>
  {
    using Ptdur = std::chrono::duration<double, std::milli>; 
    struct PerfTracker {
      std::chrono::time_point<std::chrono::high_resolution_clock> start_;
      Ptdur &ml_;

      PerfTracker(Ptdur &ml)
        : start_{std::chrono::high_resolution_clock::now()},  ml_{ml}{}
      ~PerfTracker(){
        ml_ += std::chrono::high_resolution_clock::now() - start_;
      }
    };

    Ptdur tt_m_ {};
  public:
    KmeansCpu(const Data &d, const bool random, const size_t n_clu, const unsigned max_iters)
      : KmeansBase<KmeansCpu>(d, random, n_clu, max_iters)
    {}
    Labels fit() override ;
    Centroids<double> & result() override { 
      return KmeansBase<KmeansCpu>::solved_ 
              ? KmeansBase<KmeansCpu>::c_ 
              : (fit(), KmeansBase<KmeansCpu>::c_); 
    }
    ~KmeansCpu() override {
      dbg << "CPU Time (ms): \n";
      dbg << tt_m_.count() << '\n';
    };
  }; 
}