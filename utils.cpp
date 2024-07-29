#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cassert>
#include <random>

#include "utils.hpp"
#include "args.hpp"

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

template<typename ElemType>
Labels findNearestCentroids(const Data &d, const Centroids<ElemType> &c)
{
  Labels l {};
  for(int j = 0; j < d.size(); ++j){
    const auto &pi = d[j];
    double min_d = HUGE_VAL;
    int ci = -1;
    for(int i = 0; i < c.size(); ++i) {
      const auto cd = pi.equilDist(c[i]);
      if (cd < min_d){
        ci = i;
        min_d = cd;
      }
    }
    assert(ci != -1);
    l.push_back(ci);
  }
  return l;
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

  for(int i = 0; i < l.size(); ++i){
    new_c[l[i]] += d[i];
    sizes[l[i]]++;
  }
  for(int i = 0; i < new_c.size(); ++i) new_c[i] /= sizes[i]; 
}

template<typename C1, typename C2>
bool converged(const C1 &c, const C2 &oldc)
{
  assert(c.size() == oldc.size());
  bool res = true;
  dbg << "Dist = " << std::setprecision(20);
  for(int i = 0; i < c.size(); ++i){
    const auto dist = c[i].equilDist(oldc[i]);
    dbg << dist << ' ';
    if(dist > Args::t) res &= false;
  }
  dbg << '\n';
  return res;
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
template<typename LongType> 
long double Point<ElemType>::equilDist(const Point<LongType> &p2) const {
  long double d = 0;
  assert(size() == p2.size());
  for(int i = 0; i < size(); ++i){
    long double delta = static_cast<long double>(elems_[i]) - p2.elems_[i];
    d += delta * delta;
  }
  return sqrtl(d);
}

template<typename ElemType>
template<typename LongType> 
long double Point<ElemType>::minDist(const Point<LongType> &p2) const {
  long double d = HUGE_VALL;
  assert(size() == p2.size());
  for(int i = 0; i < size(); ++i){
    long double delta = std::abs(static_cast<long double>(elems_[i]) - p2.elems_[i]);
    if(delta < d) d = delta;
  }
  return d;
}

template<typename ElemType>
DebugStream& operator<<(DebugStream &os, const Point<ElemType> &point){
  for(int i = 0; i < point.size(); ++i) os << point.elems_[i] << ' ';
  os << '\n';
  return os;
}
template
DebugStream& operator<<(DebugStream &os, const DoublePoint &point);
template
DebugStream& operator<<(DebugStream &os, const LongPoint &point);

template<typename ElemType>
std::ostream & operator<<(std::ostream &os, const Point<ElemType> &point){
  for(int i = 0; i < point.size(); ++i) os << point.elems_[i] << ' ';
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

DoubleCentroids Data::randomCentroids(const unsigned n_clu){
  DoubleCentroids c {n_clu, dim_};
  dbg << __func__ << '\n';
  for (int i=0; i<n_clu; i++){
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

template<typename ElemType>
Labels Kmeans<ElemType>::fit(){
  PerfTracker pt {tt_m_};

  Centroids<ElemType> old_c{c_};

  Labels l;
  while(true){
    // labels is a mapping from each point in the dataset 
    // to the nearest (euclidean distance) centroid
    l = findNearestCentroids(d_, old_c);

    // the new centroids are the average 
    // of all the points that map to each 
    // centroid
    averageLabeledCentroids(d_, l, old_c, c_);
    dbg << "ITER = " << iters_ << '\n';
    //print_centroids(dbg, c_);
    if(++iters_ > max_iters_ || converged(c_, old_c)) break;
    //done = ++iters_ > max_iters_;

    old_c = c_;    
  }

  solved = true;
  return l;
}
template Labels Kmeans<double>::fit();
template Labels Kmeans<long double>::fit();

} // namespace utils