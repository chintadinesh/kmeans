#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cassert>
#include "utils.hpp"
#include "args.hpp"

using namespace std;

extern utils::DebugStream dbg;

namespace {
using namespace utils;
unsigned long int _next = 1;
unsigned long _kmeans_rmax = 32767;

std::vector<double> parseDoubles(const string &line){
  std::vector<double> result;
  std::istringstream stream{line};
  double value;
  stream >> value; // the first value is just the point index
  while (stream >> value) result.push_back(value);

  return result;
}

Labels findNearestCentroids(const Data &d, const LongCentroids &c)
{
  Labels l {};
  for(int i = 0; i < d.size(); ++i){
    const auto &pi = d[i];
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

LongCentroids averageLabeledCentroids(const Data &d, const Labels &l, const LongCentroids &old_c)
{
  assert(d.size() == l.size());
  LongCentroids new_c;
  vector<unsigned> sizes = vector<unsigned>(old_c.size(), 0);

  for(const auto &ci: old_c) 
    new_c.emplace_back( vector<long double>(ci.size(),0l) );

  for(int i = 0; i < l.size(); ++i){
    new_c[l[i]] += d[i];
    sizes[l[i]]++;
  }
  for(int i = 0; i < new_c.size(); ++i) new_c[i] /= sizes[i]; 
  return new_c;
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

} // unnamed namespace

namespace utils {

unsigned kmeans_rand() {
  _next = _next * 1103515245 + 12345;
  return (unsigned int)(_next/65536) % (_kmeans_rmax+1);
}

void kmeans_srand(unsigned int seed) { _next = seed; }

template<typename ElemType>
template<typename LongType> 
long double Point<ElemType>::equilDist(const Point<LongType> &p2) const {
  long double d = 0;
  assert(size() == p2.size());
  for(int i = 0; i < dims_.size(); ++i){
    long double delta = static_cast<long double>(dims_[i]) - p2.dims_[i];
    d += delta * delta;
  }
  return sqrtl(d);
}

template<typename ElemType>
DebugStream& operator<<(DebugStream &os, const Point<ElemType> &point){
  for(const auto i: point.dims_) os << i << ' ';
  os << '\n';
  return os;
}
template
DebugStream& operator<<(DebugStream &os, const DoublePoint &point);
template
DebugStream& operator<<(DebugStream &os, const LongPoint &point);

template<typename ElemType>
std::ostream & operator<<(std::ostream &os, const Point<ElemType> &point){
  for(const auto i: point.dims_) os << i << ' ';
  os << '\n';
  return os;
}

Data parseInput(const string &input_file){
  Data result; 

  string line;

  ifstream file; ;
  file.open(input_file);
  if (!file.is_open()) {
    cerr << "Could not open the file: " << input_file << endl;
    return result;
  }
  
  getline(file, line);

  while(getline(file, line)) result.emplace_back( parseDoubles(line) );

  Args::d = result[0].size();

  file.close();
  return result;
}

DebugStream & operator<<(DebugStream &os, const Data &data){
  os << "Size = " << data.size() << '\n';
  for(const auto &p: data.points_) os << p;
  return os;
}

Centroids Data::randomCentroids(const unsigned n_clu){
  Centroids c {};
  for (int i=0; i<n_clu; i++){
    unsigned index = kmeans_rand() % size();
    c.push_back(points_[index]);
  }
  return c;
}

template<typename Stream, typename C>
Stream & print_centroids(Stream &os, const C &ctrs){
  for(unsigned i = 0; i < ctrs.size(); ++i) os << ctrs[i];
  return os;
}

template 
DebugStream & print_centroids(DebugStream &os, const Centroids &ctrs);
template 
std::ostream & print_centroids(std::ostream &os, const Centroids &ctrs);
template 
DebugStream & print_centroids(DebugStream &os, const LongCentroids &ctrs);
template 
std::ostream & print_centroids(std::ostream &os, const LongCentroids &ctrs);

Labels Problem::solve(){
  PerfTracker pt {tt_m_};

  Labels l;
  bool done = false;
  while(!done){
    LongCentroids old_c = c_;

    // labels is a mapping from each point in the dataset 
    // to the nearest (euclidean distance) centroid
    l = findNearestCentroids(d_, c_);

    // the new centroids are the average 
    // of all the points that map to each 
    // centroid
    c_ = averageLabeledCentroids(d_, l, old_c);
    dbg << "ITER = " << iters_ << '\n';
    //print_centroids(dbg, c_);
    done = ++iters_ > max_iters_ || converged(c_, old_c);
  }

  solved = true;
  return l;
}

} // namespace utils