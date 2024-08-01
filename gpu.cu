#include "gpu.hpp"

extern utils::DebugStream dbg;

namespace utils {

/* KmeansGpu */
template<typename ElemType>
Labels KmeansGpu<ElemType>::fit(){

  auto &c = KmeansBase<KmeansGpu<ElemType>, ElemType>::c_;
  auto &d = KmeansBase<KmeansGpu<ElemType>, ElemType>::d_;
  auto &solved = KmeansBase<KmeansGpu<ElemType>, ElemType>::solved_;
  auto &iters = KmeansBase<KmeansGpu<ElemType>, ElemType>::iters_;
  auto &max_iters = KmeansBase<KmeansGpu<ElemType>, ElemType>::max_iters_;

  stgy_->init(d.ptr(), 
              c.ptr(), 
              sizeof(double)*d.size()*d.dim(), 
              sizeof(double)*c.size()*c.dim());

  Labels l(d.size(), 0);

  while(true){
    // labels is a mapping from each point in the dataset 
    // to the nearest (euclidean distance) centroid
    stgy_->findNearestCentroids();

    // the new centroids are the average 
    // of all the points that map to each 
    // centroid
    stgy_->averageLabeledCentroids();
    dbg << "ITER = " << iters << '\n';
    //print_centroids(dbg, c_);
    if(++iters > max_iters || stgy_->converged()) break;
    //done = ++iters_ > max_iters_;

    stgy_->swap();
  }

  stgy_->collect(c.ptr(), 
                &l[0], 
                sizeof(double)*c.dim()*c.size(), 
                sizeof(unsigned)*d.size());

  solved = true;
  return l;
}
template Labels KmeansGpu<double>::fit();

}