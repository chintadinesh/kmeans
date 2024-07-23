#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "../inc/utils.hpp"

using namespace std;

namespace {
  utils::Point parseDoubles(const string &line){
    utils::Point result;
    std::istringstream stream{line};
    double value;

    while (stream >> value) result.push_back(value);

    return result;
  }
}

namespace utils {

Data parseInput(const string &input_file){
  Data result; 

  string line;

  ifstream file; ;
  file.open(input_file);
  if (!file.is_open()) {
    cerr << "Could not open the file: " << input_file << endl;
    return result;
  }

  while(getline(file, line)) result.push_back( parseDoubles(line) );

  file.close();
  return result;
}

ostream & operator<<(ostream &os, const Data &data){
  os << "Size = " << data.size() << '\n';
  for(const auto &v: data){
    for(const auto i: v) cout << i << ' ';
    os << '\n';
  }
  return os;
}

}