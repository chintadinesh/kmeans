#pragma once

#include <cassert>
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>


namespace utils {
  class DebugStream;

  template<typename ElemType = double>
  class Point {
    /// @brief magnitude in each dimension
    ElemType *elems_; 
    size_t size_;
  public:
    Point(ElemType *dst_ptr, size_t size) : elems_{dst_ptr}, size_{size} {}

    Point(const Point &p1) = delete;
    Point & operator=(const Point &p1) = delete;

    Point(Point &&p1) = delete;
    Point & operator=(Point &&p1) = delete;

    void copy(const Point &p) {
      assert(size_ == p.size_);
      memcpy(elems_, p.elems_, sizeof(ElemType)*size_);
    }

    Point & operator=(const std::vector<ElemType> &v) {
      memcpy(elems_, &v[0], sizeof(ElemType)*size_);
      return *this;
    }
    
#if 0
    template<typename DstElemType>
    operator Point<DstElemType>() const {
      Point<DstElemType> rp;
      for(const auto &e: elems_) rp.push_back(static_cast<DstElemType>(e));
      return rp;
    }
#endif

    ElemType equilDist(const Point &p2) const;
    ElemType minDist(const Point &p2) const;

    size_t size() const {return size_;}
    //void push_back(const  ElemType &e){elems_.push_back(e);}

    template<typename AddElemType = double>
    Point & operator+=(const Point<AddElemType> &p2){
      for(int i = 0;i < p2.size(); ++i)
        elems_[i] += p2.elems_[i];
      return *this;
    }

    Point & operator/=(const unsigned div){
      for(int i = 0;i < size(); ++i)
        elems_[i] /= div;
      return *this;
    }

    ElemType & operator[](const unsigned i){elems_[i];}
    const ElemType & operator[](const unsigned i) const {elems_[i];}

    template<typename LongType>
    friend class Point;

    template<typename T>
    friend DebugStream& operator<<(DebugStream &os, const Point<T> &point);
    template<typename T>
    friend std::ostream& operator<<(std::ostream &os, const Point<T> &point);
  };

  using LongPoint = Point<long double>;
  using DoublePoint = Point<double>;

  template<typename ElemType>
  class Centroids {
    size_t n_clu_;
    size_t dim_;
    std::unique_ptr<ElemType> elems_;
  public:
    Centroids(size_t n_clu, size_t dim) 
      : n_clu_{n_clu}, dim_{dim}, elems_{new ElemType[n_clu*dim]}
    {}

    Centroids(const Centroids<ElemType> &c) 
      : n_clu_{c.n_clu_}, dim_{c.dim_}, elems_{new ElemType[c.n_clu_*c.dim_]}
    {
      memcpy(elems_.get(), c.elems_.get(), sizeof(ElemType)*dim_*n_clu_);
    }
    Centroids & operator=(const Centroids<ElemType> &c) {
      if(&c != this){
        assert(n_clu_ == c.n_clu_);
        assert(dim_ == c.dim_);
        memcpy(elems_.get(), c.elems_.get(), sizeof(double)*n_clu_*dim_);
      }
      return *this;
    }

    Point<ElemType> operator[](const unsigned i) const {
      assert(i < n_clu_);
      return Point<ElemType>{elems_.get() + i*dim_, dim_};
    }

    const size_t size() const {return n_clu_;}

    void reset() {memset(elems_.get(), 0, sizeof(ElemType)*n_clu_*dim_);}
  };

  using DoubleCentroids = Centroids<double>;

  class Data {
    size_t dim_;
    size_t size_;
    double *all_elems_;
  public:
    explicit Data(const std::string &file_name);
    ~Data(){delete[] all_elems_;}

    const unsigned size() const {return size_;} 
    const unsigned dim() const {return dim_;} 

    const DoublePoint operator[](const unsigned i) const {
      assert(i<size_);
      return DoublePoint {&all_elems_[dim_*i], dim_};
    }
    DoublePoint operator[](const unsigned i) {
      assert(i<size_);
      return DoublePoint {&all_elems_[dim_*i], dim_};
    }

    DoubleCentroids randomCentroids(const unsigned n_clu) const;

    friend DebugStream & operator<<(DebugStream &os, const Data &data);
  };

  using Labels = std::vector<unsigned>;

  // interface class for kmeans algorithm
  template<class Concrete, typename ElemType>
  class KmeansBase {
protected:
    const Data &d_;
    Centroids<ElemType> c_;
    Centroids<ElemType> old_c_;

    const unsigned max_iters_;
    unsigned iters_ {};
    bool solved_ = false;

public:
    KmeansBase(const Data &d, 
              const bool random,
              const size_t n_clu,
              const unsigned max_iters);
    Labels fit() { static_cast<Concrete>(*this).fit(); }
    Centroids<ElemType> & result(){return static_cast<Concrete>(*this).result();}
    // I am not sure how destructors are made virtual in CRTP.
    virtual ~KmeansBase() = default;
  };

  template<typename ElemType>
  class KmeansCpu : public KmeansBase<KmeansCpu<ElemType>, ElemType>
  {
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
  public:
    KmeansCpu(const Data &d, const bool random, const size_t n_clu, const unsigned max_iters)
      : KmeansBase<KmeansCpu<ElemType>, ElemType>(d, random, n_clu, max_iters)
    {}
    Labels fit();
    Centroids<ElemType> & result(){ 
      return KmeansBase<KmeansCpu<ElemType>, ElemType>::solved_ 
              ? KmeansBase<KmeansCpu<ElemType>, ElemType>::c_ 
              : (fit(), KmeansBase<KmeansCpu<ElemType>, ElemType>::c_); 
    }
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

  DoubleCentroids randomCentroids(const unsigned n_clu, const unsigned dim);

  void kmeans_srand(unsigned int seed);
}
