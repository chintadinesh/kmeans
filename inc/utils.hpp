#pragma once

#include <vector>
#include <iostream>

namespace utils {
  class Point {
    /// @brief cluster id
    int clid_ {-1};  
    /// @brief magnitued in each dimension
    std::vector<double> dims_{}; 
  public:
    explicit Point(const std::vector<double> &pt) : dims_{pt} {}
    friend std::ostream & operator<<(std::ostream &os, const Point &point);
  };

  class Data {
    std::vector<Point> points_;
    std::vector<Point> cls_;
  public:
    unsigned size() const {return points_.size();}
    void emplace_back(const std::vector<double> &pt) {points_.emplace_back(pt);}
    void push_back(const Point &pt) {points_.push_back(pt);}

    void randomCentroid(const unsigned n_clu);

    friend std::ostream & operator<<(std::ostream &os, const Data &data);
  };

  Data parseInput(const std::string &input_file);
}