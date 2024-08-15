#include <fstream>

#include "stats.hpp"

namespace kmeans {

Stats & Stats::instance() {
  static Stats inst;
  return inst;
}

void Stats::record(const Event *ev) {
  event_times_.emplace_back(ev->clone());
}

Stats::~Stats(){
  std::ofstream os {"time_stats.rpt"};
  os << "Event Times: \n";
  float total = 0;
  for(const auto &evp: event_times_){
    evp->print(os);
    total += evp->count_ms();
  }
  os << "Total = " << total << '\n';
}
}