#pragma once

#include <ostream>
#include <vector>

namespace kmeans {
  class Event {
  public:
    virtual ~Event() = 0; 
    virtual float count_ms() const = 0;
    virtual std::ostream & print(std::ostream &os) const = 0;
  };
  inline Event::~Event() {}

  class Stats{
  public:
  private:
    static inline std::vector<const Event *> event_times_ {};

  public:
    static void record(const Event *ev) {event_times_.push_back(ev);}

    ~Stats();
  };
}