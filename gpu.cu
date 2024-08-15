#include "gpu.hpp"
#include "args.hpp"
#include "stats.hpp"

extern kmeans::DebugStream dbg;

namespace {

__global__ void classify(unsigned *l,
                        const double *d, 
                        const double *c, 
                        const size_t d_sz,
                        const size_t c_sz,
                        const size_t dim)
{
  size_t tid = blockDim.x*blockIdx.x + threadIdx.x;

  if(d_sz <= tid) return;

  const double *point = &d[tid*dim];

  unsigned res = 0;
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

__global__ void update(double *tmp_c, 
                      unsigned *tmp_npts,
                      const double *d, 
                      const unsigned *l,
                      const size_t c_sz,
                      const size_t d_sz,
                      const size_t dim)
{
  extern __shared__ char sh_all[];

  // set the counter and centroids locations in the shared memory
  unsigned *npts = (unsigned *)(&sh_all[0]);
  double *cent = (double *)(sh_all + c_sz*sizeof(unsigned));

  // initialize the shared memory
  if(threadIdx.x == 0){ // thread 0 of all blocks initializes it's shared mem
    for(size_t cid = 0; cid < c_sz; ++cid){
      npts[cid] = 0;
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
    double *c = &tmp_c[blockIdx.x*c_sz*dim];
    unsigned *pts = &tmp_npts[blockIdx.x*c_sz];
    for(size_t cid = 0; cid < c_sz; ++cid){
      pts[cid] = npts[cid]; // update the number of pts in the centroid
      for(size_t dimid = 0; dimid < dim; ++dimid){ // update the centroid
        c[cid*dim + dimid] = cent[cid*dim + dimid]; 
      }
    }
  }
}

// use with only one block
__global__ void reduce(double *c, const double *tmp_c, const unsigned *npts, const unsigned c_sz, const unsigned dim){
  // shared memory for local accumulation
  extern __shared__ char sh_all[];

  // set the counter and centroids locations in the shared memory
  unsigned *sh_npts = (unsigned *)(&sh_all[0]);
  double *sh_c = (double *)(sh_all + blockDim.x*c_sz*sizeof(unsigned));

  unsigned tid = threadIdx.x;

  // initialize the shared memeory with partial centroids
  for(unsigned cid = 0; cid < c_sz; ++cid){
    sh_npts[tid*c_sz + cid] = npts[tid*c_sz + cid];
    for(unsigned did = 0; did < dim; ++did){
      sh_c[tid*c_sz*dim + cid*dim + did] = tmp_c[tid*c_sz*dim + cid*dim + did];
    }
  }

  __syncthreads();
  
  // Perform reduction in shared memory
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      for(unsigned cid = 0; cid < c_sz; ++cid){
        sh_npts[tid*c_sz + cid] += sh_npts[(tid + s)*c_sz + cid];
        for(unsigned did = 0; did < dim; ++did){
          sh_c[tid*c_sz*dim + cid*dim + did] += sh_c[(tid + s)*c_sz*dim + cid*dim + did];
        }
      }
    }
    __syncthreads();
  }

  // copy the centroid to global memory
  if(tid == 0){
    for(unsigned cid = 0; cid < c_sz; ++cid){
      for(unsigned did = 0; did < dim; ++did){
        c[cid*dim +  did] = sh_c[cid*dim + did]/sh_npts[cid];
      }
    }
  }
}

  class GPUEvent: public kmeans::Event {
  private:
    class GPUTimer{
      inline static cudaEvent_t start_ {}, end_ {};

      GPUTimer(){
        cudaEventCreate(&start_);
        cudaEventCreate(&end_);
      }

    public:
      static GPUTimer & inst(){
        static GPUTimer instance; 
        return instance;
      }

      ~GPUTimer(){
        cudaEventDestroy(start_);
        cudaEventDestroy(end_); 
      }

      static void start() { cudaEventRecord(start_, 0); }

      static float end() {
        cudaEventRecord(end_, 0); 
        cudaEventSynchronize(end_);
        float elapsed_time;
        cudaEventElapsedTime(&elapsed_time, start_, end_);
        return elapsed_time;
      }
    };

  public:
    enum EventType {MEMCPY = 0, CLASSIFY, UPDATE, REDUCE, OTHERS, _EVENT_TYPE_LEN};

    explicit GPUEvent(const EventType ev) : ev_{ev} { GPUTimer::inst().start(); }

    GPUEvent * clone() const override {return new GPUEvent{*this};}

    ~GPUEvent() {
      auto tm = GPUTimer::inst().end();
      elapsed_time_ = tm;
      kmeans::Stats::instance().record(this);
    }

    float count_ms() const override {return elapsed_time_;}

    std::ostream & print(std::ostream &os) const override {
      os << "\t\t {" << event_names[ev_] << ": " << elapsed_time_ << "}";
      return os;
    }

  private:
    static const std::string event_names[_EVENT_TYPE_LEN];

    EventType ev_;
    float elapsed_time_ {};
  };

  const std::string GPUEvent::event_names[] = {
    "MEMCPY", 
    "CLASSIFY", 
    "UPDATE", 
    "REDUCE",
    "OTHERS", 
  };

} // anonymous namespace


namespace kmeans {

KmeansStrategyGpuGlobalBase::KmeansStrategyGpuGlobalBase(const size_t sz, const size_t k, const size_t dim)
  : d_sz_{sz}, c_sz_{k}, dim_{dim}
{
  gpuErrchk( cudaMalloc((void**)&data_device_, sizeof(double)*dim*sz) );
  gpuErrchk( cudaMalloc((void**)&c_device_, sizeof(double)*dim*k) );
  gpuErrchk( cudaMalloc((void**)&old_c_device_, sizeof(double)*dim*k) );
  gpuErrchk( cudaMalloc((void**)&labels_device_, sizeof(unsigned)*sz) );

  // temporary storage before the centroids are accumulated
  gpuErrchk( cudaMalloc((void**)&tmp_c_device_, sizeof(double)*dim*k*Args::blocks_update) );
  gpuErrchk( cudaMalloc((void**)&tmp_npts_device_, sizeof(unsigned)*k*Args::blocks_update) );

}

void KmeansStrategyGpuGlobalBase::init(const double *d, const double *c, const size_t data_sz, const size_t c_sz) {
  {
    GPUEvent ev{GPUEvent::MEMCPY}; // RAII
    gpuErrchk( cudaMemcpy(data_device_, d, data_sz, cudaMemcpyHostToDevice) );
  }
  if(Args::debug){
    std::vector<double> d_test(data_sz/sizeof(double), 0);
    gpuErrchk( cudaMemcpy(&d_test[0], data_device_, data_sz, cudaMemcpyDeviceToHost) );
    for(int i = 0; i < data_sz/sizeof(double); ++i) assert(d_test[i] == d[i]);
  }

  {
    GPUEvent ev{GPUEvent::MEMCPY}; // RAII
    gpuErrchk( cudaMemcpy(c_device_, c, c_sz, cudaMemcpyHostToDevice) );
  }
  if(Args::debug){
    std::vector<double> c_test(c_sz/sizeof(double), 0);
    gpuErrchk( cudaMemcpy(&c_test[0], c_device_, c_sz, cudaMemcpyDeviceToHost) );
    for(int i = 0; i < c_sz/sizeof(double); ++i) assert(c_test[i] == c[i]);
  }

  {
    GPUEvent ev{GPUEvent::MEMCPY}; // RAII
    gpuErrchk( cudaMemcpy(old_c_device_, c, c_sz, cudaMemcpyHostToDevice) );
  }
}

void KmeansStrategyGpuGlobalBase::collect(double *c, unsigned *l, const size_t c_sz, const size_t l_sz) {
  {
    GPUEvent ev{GPUEvent::MEMCPY}; // RAII
    gpuErrchk( cudaMemcpy(c, c_device_, c_sz, cudaMemcpyDeviceToHost) );
  }

  {
    GPUEvent ev{GPUEvent::MEMCPY}; // RAII
    gpuErrchk( cudaMemcpy(l, labels_device_, d_sz_, cudaMemcpyDeviceToHost) );
  }
}

void KmeansStrategyGpuGlobalBase::swap() { std::swap(c_device_, old_c_device_); }

KmeansStrategyGpuGlobalBase::~KmeansStrategyGpuGlobalBase() {
  gpuErrchk( cudaFree(data_device_) );
  gpuErrchk( cudaFree(c_device_) );
  gpuErrchk( cudaFree(old_c_device_) );
  gpuErrchk( cudaFree(tmp_npts_device_) );
  gpuErrchk( cudaFree(labels_device_) );
  gpuErrchk( cudaFree(tmp_c_device_) );
}

void KmeansStrategyGpuGlobalBase::getCentroids(double *host_c) {
  dbg << __func__ << '\n';
  {
    GPUEvent ev{GPUEvent::MEMCPY}; // RAII
    gpuErrchk( cudaMemcpy(host_c, c_device_, sizeof(double)*c_sz_*dim_, cudaMemcpyDeviceToHost) );
  }
}

void KmeansStrategyGpuBaseline::findNearestCentroids() {
  const unsigned NTHREADS = Args::threads_classify;
  const unsigned NBLOCKS = Args::blocks_classify;

  dbg << "classify<<< " << NBLOCKS << ", " << NTHREADS << " >>>()\n";
  {
    GPUEvent ev{GPUEvent::CLASSIFY}; // RAII
    classify<<<NTHREADS, NBLOCKS>>>(labels_device_, 
                          data_device_, 
                          old_c_device_, 
                          d_sz_, 
                          c_sz_, 
                          dim_);   
  }
  gpuErrchk( cudaPeekAtLastError() );

  if(Args::debug){
    static std::unique_ptr<unsigned> labels {new unsigned[d_sz_]};
    cudaMemcpy(labels.get(), labels_device_, sizeof(unsigned)*d_sz_, cudaMemcpyDeviceToHost);
    dbg << "Labels: ";
    for(int i = 0; i < d_sz_; ++i) dbg << labels.get()[i] << ' ';
    dbg << '\n';
  }
}

void KmeansStrategyGpuBaseline::averageLabeledCentroids() 
{
  const unsigned NTHREADS = Args::threads_update;
  const unsigned NBLOCKS = Args::blocks_update;

  dbg <<  "update<<< " << NBLOCKS << ", " << NTHREADS << " >>>()\n";
  {
    GPUEvent ev{GPUEvent::UPDATE}; // RAII
    update<<<NBLOCKS, 
            NTHREADS, 
            sizeof(unsigned)*c_sz_ + sizeof(double)*c_sz_*dim_
            >>> (tmp_c_device_, 
                tmp_npts_device_,
                data_device_, 
                labels_device_, 
                c_sz_, 
                d_sz_, 
                dim_);
    gpuErrchk( cudaPeekAtLastError() );

    cudaDeviceSynchronize();
    gpuErrchk( cudaPeekAtLastError() );
  }

  // one thread is enough 
  dbg <<  "reduce<<< " << 1 << ", " << NBLOCKS << " >>>()\n";
  {
    GPUEvent ev{GPUEvent::REDUCE}; // RAII
    reduce<<<1, 
      NBLOCKS,    // this is intentional
      sizeof(unsigned)*c_sz_*NBLOCKS + sizeof(double)*c_sz_*dim_*NBLOCKS
      >>> ( c_device_, 
            tmp_c_device_, 
            tmp_npts_device_, 
            c_sz_, 
            dim_);
  }
  gpuErrchk( cudaPeekAtLastError() );

  {
    DoubleCentroids tmp_c{c_sz_, dim_};
    gpuErrchk( cudaMemcpy(tmp_c.ptr(), c_device_, c_sz_*dim_, cudaMemcpyDeviceToHost) );
    dbg << "Updated Centroids: \n";
    print_centroids(dbg, tmp_c);
  }

}

/* KmeansGpu */
Labels KmeansGpu::fit(){

  dbg << __func__ << '\n';

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
    stgy_->getCentroids(c.ptr());
    if(++iters > max_iters || converged(c.ptr(), old_c.ptr(), c.size(), c.dim())) break;
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
