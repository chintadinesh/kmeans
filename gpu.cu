#include "gpu.hpp"

extern utils::DebugStream dbg;

namespace {

__global__ void classify(size_t *l,
                        const double *d, 
                        const double *c, 
                        const size_t d_sz,
                        const size_t c_sz,
                        const size_t dim)
{
  size_t tid = blockDim.x*blockIdx.x + threadIdx.x;

  if(d_sz <= tid) return;

  const double *point = &d[tid*dim];

  size_t res = 0;
  double min_d = HUGE_VAL;
  for(size_t cl = 0; cl < c_sz; ++cl){
    double diff = 0;
    const double *cluster = &c[cl*dim];
    for(size_t di = 0; di < dim; ++di) 
      diff += (point[di] - cluster[di])*(point[di] - cluster[di]);
    if(diff < min_d){
      min_d = diff;
      res = cl;
    }
  }
  l[tid] = res;
}

__device__ double atomicAddDouble(double* address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
      assumed = old;
      old = atomicCAS(address_as_ull, assumed,
                      __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);

  return __longlong_as_double(old);
}

__global__ void update(double *c, 
                      const double *d, 
                      const size_t *l,
                      const size_t c_sz,
                      const size_t d_sz,
                      const size_t dim)
{
  extern __shared__ size_t sh_all[];

  // set the counter and centroids locations in the shared memory
  int *npts = (int *)(&sh_all[0]);
  double *cent = (double *)(&sh_all[c_sz]);

  // initialize the shared memory
  if(threadIdx.x == 0){
    for(size_t i = 0; i < dim; ++i){
      npts[i] = 0;
    }
    for(size_t cid = 0; cid < c_sz; ++cid){
      for(size_t dimid = 0; dimid < dim; ++dimid){
        cent[cid*dim + dimid] = 0;
      }
    }
  }

  __syncthreads(); // wait all threads until the initialization completes

  size_t tid = blockIdx.x*blockDim.x + threadIdx.x;
  atomicAdd(&npts[l[tid]], 1);
  for(size_t dimid = 0; dimid < dim; ++dimid)
    atomicAddDouble(&cent[l[tid]*dim + dimid], d[tid*dim + dimid]);

  __syncthreads(); // After all threads accumulate

  if(threadIdx.x == 0){
    for(size_t cid = 0; cid < c_sz; ++cid)
      for(size_t dimid = 0; dimid < dim; ++dimid){
        cent[cid*dim + dimid] /= npts[cid]; // div with num of pts in the cluster
        c[cid*dim + dimid] = cent[cid*dim + dimid]; // update centroids
      }
  }
}

}

namespace utils {

bool KmeansStrategyGpuGlobalBase::converged(double *host_c, double *host_old_c) 
{
  gpuErrchk( cudaMemcpy(host_c, c_device_, c_sz_, cudaMemcpyDeviceToHost) );
  return utils::converged(host_c, host_old_c, c_sz_, dim_);
}

void KmeansStrategyGpuBaseline::findNearestCentroids() 
{
  classify<<<1, d_sz_>>>(labels_, 
                        data_device_, 
                        old_c_device_, 
                        d_sz_, 
                        c_sz_, 
                        dim_);   
}

void KmeansStrategyGpuBaseline::averageLabeledCentroids() 
{
  update<<<1, 
          d_sz_, 
          sizeof(int)*c_sz_ + sizeof(double)*c_sz_*dim_>>>
          (c_device_,
            data_device_,
            labels_,
            c_sz_,
            d_sz_,
            dim_);
}

/* KmeansGpu */
Labels KmeansGpu::fit(){

  auto &c = KmeansBase<KmeansGpu>::c_;
  auto &old_c = KmeansBase<KmeansGpu>::old_c_;
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
    if(++iters > max_iters || stgy_->converged(c.ptr(), old_c.ptr())) break;
    //done = ++iters_ > max_iters_;

    stgy_->swap();
    std::swap(c, old_c);
  }

  stgy_->collect(c.ptr(), 
                &l[0], 
                sizeof(double)*c.dim()*c.size(), 
                sizeof(unsigned)*d.size());

  solved = true;
  return l;
}

}