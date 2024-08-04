#include <cmath>

#include "cpu.hpp"
#include "utils.hpp"
#include "args.hpp"

using namespace utils;
using namespace std;

extern utils::DebugStream dbg;

namespace {

template<typename ElemType>
void findNearestCentroids(Labels &l, const Data &d, const Centroids<ElemType> &c)
{
  for(unsigned j = 0; j < d.size(); ++j){
    const auto &pi = d[j];
    double min_d = HUGE_VAL;
    int ci = -1;
    for(unsigned i = 0; i < c.size(); ++i) {
      const auto cd = pi.equilDist(c[i]);
      if (cd < min_d){
        ci = i;
        min_d = cd;
      }
    }
    assert(ci != -1);
    l[j] = ci;
  }
}

template<typename ElemType>
void averageLabeledCentroids(const Data &d, 
                            const Labels &l, 
                            const Centroids<ElemType> &old_c,
                            Centroids<ElemType> &new_c)
{
  assert(d.size() == l.size());
  new_c.reset();

  vector<unsigned> sizes = vector<unsigned>(old_c.size(), 0);

  for(unsigned i = 0; i < l.size(); ++i){
    new_c[l[i]] += d[i];
    sizes[l[i]]++;
  }
  for(unsigned i = 0; i < new_c.size(); ++i) new_c[i] /= sizes[i]; 
}

}

namespace utils 
{

inline bool converged(const Centroids<double> &c, const Centroids<double> &old_c){
  return converged(c.ptr(), old_c.ptr(), c.size(), c.dim());
}


Labels KmeansCpu::fit(){

  auto &c = KmeansBase<KmeansCpu>::c_;
  auto &old_c = KmeansBase<KmeansCpu>::old_c_;
  auto &d = KmeansBase<KmeansCpu>::d_;
  auto &solved = KmeansBase<KmeansCpu>::solved_;
  auto &iters = KmeansBase<KmeansCpu>::iters_;
  auto &max_iters = KmeansBase<KmeansCpu>::max_iters_;

  old_c = c;

  Labels l(d.size(),0);

  {
    PerfTracker pt {tt_m_};
    while(true){
      // labels is a mapping from each point in the dataset 
      // to the nearest (euclidean distance) centroid
      findNearestCentroids(l, d, old_c);

      // the new centroids are the average 
      // of all the points that map to each 
      // centroid
      averageLabeledCentroids(d, l, old_c, c);
      dbg << "ITER = " << iters << '\n';
      //print_centroids(dbg, c_);
      if(++iters > max_iters || converged(c, old_c)) break;
      //done = ++iters_ > max_iters_;

      std::swap(old_c, c);    
    }
  }

  solved = true;
  return l;
}


}