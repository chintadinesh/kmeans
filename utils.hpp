#pragma once

#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>


namespace utils {
  class DebugStream;

  template<typename ElemType = double>
  class Point {
    /// @brief magnitued in each dimension
    std::vector<ElemType> dims_{}; 
  public:
    Point(){}
    explicit Point(const std::vector<ElemType> &pt) : dims_{pt} {}
    
    template<typename DstElemType>
    operator Point<DstElemType>() const {
      Point<DstElemType> rp;
      for(const auto &e: dims_) rp.push_back(static_cast<DstElemType>(e));
      return rp;
    }

    template<typename LongType>
    long double equilDist(const Point<LongType> &p2) const;
    template<typename LongType>
    long double minDist(const Point<LongType> &p2) const;

    unsigned size() const {return dims_.size();}
    void push_back(const  ElemType &e){dims_.push_back(e);}

    template<typename AddElemType = double>
    Point & operator+=(const Point<AddElemType> &p2){
      for(int i = 0;i < p2.dims_.size(); ++i)
        dims_[i] += p2.dims_[i];
      return *this;
    }

    Point & operator/=(const unsigned div){
      for(int i = 0;i < dims_.size(); ++i)
        dims_[i] /= div;
      return *this;
    }

    template<typename LongType>
    friend class Point;

    template<typename T>
    friend DebugStream& operator<<(DebugStream &os, const Point<T> &point);
    template<typename T>
    friend std::ostream& operator<<(std::ostream &os, const Point<T> &point);
  };

  using LongPoint = Point<long double>;
  using DoublePoint = Point<double>;

  using Centroids = std::vector<DoublePoint>;
  using LongCentroids = std::vector<LongPoint>;

  class Data {
    std::vector<DoublePoint> points_;
  public:
    unsigned size() const {return points_.size();}

    void emplace_back(const std::vector<double> &pt) {points_.emplace_back(pt);}

    void push_back(const DoublePoint &pt) {points_.push_back(pt);}

    const DoublePoint & operator[](const unsigned i) const {return points_[i];}
    DoublePoint & operator[](const unsigned i) {return points_[i];}

    Centroids randomCentroids(const unsigned n_clu);
    Centroids randomCentroidsExcl(const unsigned n_clu);

    friend DebugStream & operator<<(DebugStream &os, const Data &data);
  };

  using Labels = std::vector<unsigned>;

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
    LongCentroids c_;
    const unsigned max_iters_;
    /// @brief  number of iterations took for solving
    unsigned iters_ {};
    bool solved = false;

  public:
    Problem(const Data &d, const LongCentroids &c, const unsigned max_iters)
      : d_{d}, c_{c}, max_iters_{max_iters} {}
    Labels solve();
    LongCentroids result(){ return solved ? c_ : (solve(), c_); }

  };
  class DebugStream {
  public:
      DebugStream() : enabled_{false} {}
      explicit DebugStream(const std::string &fl_name) 
        : enabled_{true}, out_{new std::ofstream(fl_name)} {}

      void setEnabled(const bool enable) { enabled_ = enable; }
      // Set output stream
      void setOutputStream(std::ostream& os) { out_.reset(&os); }

      template<typename T>
      DebugStream& operator<<(const T& msg) {
          if (enabled_) (*out_) << msg;
          return *this;
      }

      // Specialization for std::endl
      DebugStream& operator<<(std::ostream& (*manip)(std::ostream&)) {
          if (enabled_) (*out_) << manip;
          return *this;
      }

  private:
      bool enabled_ {};
      std::unique_ptr<std::ostream> out_ {nullptr};
  };

  template<typename Stream, typename C>
  Stream & print_centroids(Stream &os, const C &ctrs);

  Data parseInput(const std::string &input_file);

  Centroids randomCentroids(const unsigned n_clu, const unsigned dim);

  void kmeans_srand(unsigned int seed);
}
