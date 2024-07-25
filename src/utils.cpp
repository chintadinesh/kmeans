#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cassert>
#include "../inc/utils.hpp"
#include "../inc/args.hpp"

using namespace std;
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

Labels findNearestCentroids(const Data &d, const Centroids &c)
{
  Labels l {};
  for(int i = 0; i < d.size(); ++i){
    const auto &pi = d[i];
    double min_d = HUGE_VAL;
    int ci = -1;
    for(int i = 0; i < c.size(); ++i) {
      const auto cd = pi.equilDist(c[ci]);
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

Centroids averageLabeledCentroids(const Data &d, const Labels &l, const Centroids &old_c)
{
  assert(d.size() == l.size());
  Centroids new_c;
  vector<unsigned> sizes = vector<unsigned>(old_c.size(), 0);
  for(const auto &ci: old_c) 
    new_c.emplace_back( vector<double>(ci.size(),0l) );
  for(int i = 0; i < l.size(); ++i){
    new_c[l[i]] += d[i];
    sizes[l[i]]++;
  }
  for(int i = 0; i < new_c.size(); ++i) new_c[i] /= sizes[i]; 
  return new_c;
}

bool converged(const Centroids &c, const Centroids &oldc)
{
  assert(c.size() == oldc.size());
  for(int i = 0; i < c.size(); ++i) if(c[i].equilDist(oldc[i]) > Args::t) return false;
  return true;
}

} // unnamed namespace

namespace utils {

unsigned kmeans_rand() {
  _next = _next * 1103515245 + 12345;
  return (unsigned int)(_next/65536) % (_kmeans_rmax+1);
}

void kmeans_srand(unsigned int seed) { _next = seed; }

double Point::equilDist(const Point &p2) const {
  double d = 0l;
  cout <<  "INFO: " << dims_.size() << ' ' << p2.dims_.size() << '\n';
  assert(dims_.size() == p2.dims_.size());
  for(int i = 0; i < dims_.size(); ++i) d += (dims_[i] - p2.dims_[i])*(dims_[i] - p2.dims_[i]);
  return sqrtl(d);
}

std::ostream & operator<<(std::ostream &os, const Point &point){
  for(const auto i: point.dims_) cout << i << ' ';
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
  int np = stoi(line);

  while(getline(file, line)) result.emplace_back( parseDoubles(line) );

  Args::d = result[0].size();

  file.close();
  return result;
}

ostream & operator<<(ostream &os, const Data &data){
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

ostream & print_centroids(std::ostream &os, const Centroids &ctrs) {
  for(unsigned i = 0; i < ctrs.size(); ++i) os << i << ' ' << ctrs[i];
  return os;
}

void print_centroids(const Centroids &ctrs) {
  print_centroids(std::cout, ctrs);
}

Centroids Problem::solve(){
  PerfTracker pt {tt_m_};

  bool done = false;
  while(!done){
    auto old_c = c_;

    // labels is a mapping from each point in the dataset 
    // to the nearest (euclidean distance) centroid
    auto labels = findNearestCentroids(d_, c_);

    // the new centroids are the average 
    // of all the points that map to each 
    // centroid
    c_ = averageLabeledCentroids(d_, labels, old_c);
    cout << "ITER = " << iters_ << '\n';
    print_centroids(c_);
    done = iters_ > max_iters_ || converged(c_, old_c);
  }

  solved = true;
  return c_;
}

}