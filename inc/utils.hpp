#pragma once

#include <vector>
#include <iostream>
#include <chrono>

namespace utils {
  class Point {
    /// @brief magnitued in each dimension
    std::vector<double> dims_{}; 
  public:
    explicit Point(const std::vector<double> &pt) : dims_{pt} {}
    double equilDist(const Point &p2) const;
    Point & operator+=(const Point &p2){
      for(int i = 0;i < p2.dims_.size(); ++i)
        dims_[i] += p2.dims_[i];
      return *this;
    }
    Point & operator/=(const unsigned div){
      for(int i = 0;i < dims_.size(); ++i)
        dims_[i] /= div;
      return *this;
    }

    friend std::ostream & operator<<(std::ostream &os, const Point &point);
  };

  using Centroids = std::vector<Point>;

  class Data {
    std::vector<Point> points_;
  public:
    unsigned size() const {return points_.size();}
    void emplace_back(const std::vector<double> &pt) {points_.emplace_back(pt);}
    void push_back(const Point &pt) {points_.push_back(pt);}
    const Point & operator[](const unsigned i) const {return points_[i];}
    Point & operator[](const unsigned i) {return points_[i];}

    Centroids randomCentroids(const unsigned n_clu);

    friend std::ostream & operator<<(std::ostream &os, const Data &data);
  };

  class Problem {
    using Ptdur = std::chrono::duration<double, std::milli>; 
    struct PerfTracker {
      std::chrono::time_point<std::chrono::high_resolution_clock> start_;
      std::chrono::time_point<std::chrono::high_resolution_clock> end_;
      Ptdur &ml_;

      PerfTracker(Ptdur &ml)
        : start_{std::chrono::high_resolution_clock::now()},  ml_{ml}{}
      ~PerfTracker(){
        ml_ += std::chrono::high_resolution_clock::now() - end_;
      }
    };

    Ptdur tt_m_ {};
    const Data &d_;
    Centroids c_;
    const unsigned max_iters_;
    const unsigned k_;
    /// @brief  number of iterations took for solving
    unsigned iters_ {};
    bool solved = false;

  public:
    Problem(const Data &d, const Centroids &c, const unsigned max_iters, const unsigned k)
      : d_{d}, c_{c}, max_iters_{max_iters}, k_{k} {}
    Centroids solve();
    Centroids result(){ return solved ? c_ : solve(); }

  };

  using Labels = std::vector<unsigned>;
  std::ostream & print_centroids(std::ostream &os, const Centroids &ctrs);
  void print_centroids(const Centroids &ctrs);

  Data parseInput(const std::string &input_file);
}
