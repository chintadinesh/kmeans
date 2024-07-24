#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "../inc/utils.hpp"
#include "utils.hpp"

using namespace std;

namespace {
  std::vector<double> parseDoubles(const string &line){
    std::vector<double> result;
    std::istringstream stream{line};
    double value;

    while (stream >> value) result.push_back(value);

    return result;
  }

  unsigned long int _next = 1;
  unsigned long _kmeans_rmax = 32767;
}

namespace utils {

unsigned kmeans_rand() {
  _next = _next * 1103515245 + 12345;
  return (unsigned int)(_next/65536) % (_kmeans_rmax+1);
}

void kmeans_srand(unsigned int seed) { _next = seed; }

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

  file.close();
  return result;
}

ostream & operator<<(ostream &os, const Data &data){
  os << "Size = " << data.size() << '\n';
  for(const auto &p: data.points_) cout << p;
  return os;
}

void Data::randomCentroid(const unsigned n_clu){
  for (int i=0; i<n_clu; i++){
    unsigned index = kmeans_rand() % size();
    cls_.push_back(points_[index]);
  }
}

}