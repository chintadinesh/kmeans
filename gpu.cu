#include "gpu.hpp"

extern utils::DebugStream dbg;

namespace utils {

/* KmeansGpu */
Labels KmeansGpu::fit(){

  auto &c = KmeansBase<KmeansGpu>::c_;
  auto &d = KmeansBase<KmeansGpu>::d_;
  auto &solved = KmeansBase<KmeansGpu>::solved_;
  auto &iters = KmeansBase<KmeansGpu>::iters_;
  auto &max_iters = KmeansBase<KmeansGpu>::max_iters_;

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

}