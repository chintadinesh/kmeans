#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cassert>
#include <random>

#include "utils.hpp"
#include "args.hpp"
#include "gpu.hpp"
#include "cpu.hpp"

using namespace std;

extern utils::DebugStream dbg;

namespace {
using namespace utils;
unsigned long int _next = 1;
unsigned long _kmeans_rmax = 32767;

double * parseDoubles(const string &line, double *dst_ptr){
  std::istringstream stream{line};
  double value;
  stream >> value; // the first value is just the point index
  while (stream >> value) {
    *dst_ptr = value;
    dst_ptr++;
  }
  return dst_ptr;
}

unsigned kmeans_rand() {
  _next = _next * 1103515245 + 12345;
  return (unsigned int)(_next/65536) % (_kmeans_rmax+1);
}

} // unnamed namespace

namespace utils {

void kmeans_srand(unsigned int seed) { _next = seed; }

DoubleCentroids randomCentroids(const unsigned n_clu, const unsigned dim){
  std::mt19937 gen(_next); // Seed the generator
  std::uniform_real_distribution<> dis(0.0, 1.0); // Define the range

  DoubleCentroids res{n_clu, dim};
  for(size_t i = 0; i < n_clu; ++i){
    std::vector<double> vec(dim);
    for (size_t i = 0; i < dim; ++i) vec[i] = dis(gen); 
    res[i] = vec;
  }
  return res;
}

/* Point */
template<typename ElemType>
ElemType Point<ElemType>::equilDist(const Point &p2) const {
  ElemType d = 0;
  assert(size() == p2.size());
  for(unsigned i = 0; i < size(); ++i){
    double delta = elems_[i] - p2.elems_[i];
    d += delta * delta;
  }
  return sqrt(d);
}
template double Point<double>::equilDist(const Point<double> &p2) const;

template<typename ElemType>
ElemType Point<ElemType>::minDist(const Point &p2) const {
  double d = HUGE_VALL;
  assert(size() == p2.size());
  for(int i = 0; i < size(); ++i){
    double delta = std::abs(elems_[i] - p2.elems_[i]);
    if(delta < d) d = delta;
  }
  return d;
}

template<typename ElemType>
DebugStream& operator<<(DebugStream &os, const Point<ElemType> &point){
  for(size_t i = 0; i < point.size(); ++i) os << point.elems_[i] << ' ';
  os << '\n';
  return os;
}
template
DebugStream& operator<<(DebugStream &os, const DoublePoint &point);
template
DebugStream& operator<<(DebugStream &os, const LongPoint &point);

template<typename ElemType>
std::ostream & operator<<(std::ostream &os, const Point<ElemType> &point){
  for(size_t i = 0; i < point.size(); ++i) os << point.elems_[i] << ' ';
  os << '\n';
  return os;
}

/* Data */

Data::Data(const string &input_file)
  : dim_{Args::d}
{
  string line;

  ifstream file; ;
  file.open(input_file);
  if (!file.is_open()) {
    cerr << "Could not open the file: " << input_file << endl;
    throw "Bad file";
  }
  
  file >> size_;  
  all_elems_ = new double[size_ * dim_];
  auto ptr = all_elems_;
  //dbg << __func__ << ": size = " << size_ << " dim = " << dim_ << '\n';

  //dbg << "\t Parsing lines: \n";
  while(getline(file, line)){
    if(line == "") continue;

    auto old_ptr = ptr;
    //dbg << "\t line = " << line << '\n';
    ptr = parseDoubles(line, ptr); 
    //dbg << "\t old ptr = " << old_ptr << " new ptr = " << ptr << '\n'; 
    assert(ptr == old_ptr + dim_);
  } 

  file.close();
}

bool converged(double *c, double  *oldc, const size_t c_sz, const size_t dim)
{
  bool res = true;
  for(unsigned i = 0; i < c_sz; ++i){
    Point<double> p {&c[i*dim], dim};
    Point<double> old_p {&oldc[i*dim], dim};
    const auto dist = old_p.equilDist(p);
    dbg << dist << ' ';
    if(dist > Args::t) res &= false;
  }
  dbg << '\n';
  return res;
}

DoubleCentroids Data::randomCentroids(const unsigned n_clu) const {
  DoubleCentroids c {n_clu, dim_};
  dbg << __func__ << '\n';
  for (unsigned i=0; i<n_clu; i++){
    unsigned index = kmeans_rand() % size();
    c[i].copy(operator[](index));
  }
  return c;
}

#if 0
Centroids Data::randomCentroidsExcl(const unsigned n_clu){
  Centroids c {};
  for (int i=0; i<n_clu; i++){
    while(1){
      unsigned index = kmeans_rand() % size();
      double dist = HUGE_VAL;
      for(const auto &p: c){
        auto tmp = points_[index].equilDist(p);
        if(tmp<dist) dist = tmp;    // lowest distance
        if(dist < 0.01l) break;  // this point already in cluster
      }

      if(dist < 0.01l) continue;

      c.push_back(points_[index]);
      break;
    }
  }
  return c;
}
#endif

DebugStream & operator<<(DebugStream &os, const Data &data){
  os << "Size = " << data.size() << '\n';
  for(size_t i = 0; i < data.size(); ++i) os << data[i];
  return os;
}


template<typename Stream, typename C>
Stream & print_centroids(Stream &os, const C &ctrs){
  for(size_t i = 0; i < ctrs.size(); ++i) os << ctrs[i];
  return os;
}

template 
DebugStream & print_centroids(DebugStream &os, const DoubleCentroids &ctrs);
template 
std::ostream & print_centroids(std::ostream &os, const DoubleCentroids &ctrs);



template<class Concrete>
KmeansBase<Concrete>::KmeansBase(const Data &d, 
                                const bool random,
                                const size_t n_clu,
                                const unsigned max_iters)
  : d_{d}, c_{n_clu, d.dim()}, old_c_{n_clu, d.dim()}, max_iters_{max_iters} 
{
  c_ = random ? randomCentroids(n_clu, d.dim()) 
              : d.randomCentroids(n_clu);
  old_c_ = c_;
  dbg << "########### Initial Centroids\n";
  print_centroids(dbg, c_);

  {
    DebugStream init_cent {"initial.txt"};
    print_centroids(init_cent, c_);
  }
}
template 
KmeansBase<KmeansCpu>::KmeansBase(const Data &d, 
                                const bool random,
                                const size_t n_clu,
                                const unsigned max_iters);

Kmeans * kmeansFactory( const Data &d, const bool random, const size_t n_clu, const unsigned max_iters){
  if(Args::gpu) return new KmeansGpu{d, random, n_clu, max_iters}; 
  else return new KmeansCpu{d, random, n_clu, max_iters}; 
}

} // namespace utils