#include <fstream>

#include "stats.hpp"
#include "utils.hpp"

extern kmeans::DebugStream dbg;

namespace kmeans {

Stats & Stats::instance() {
  static Stats inst;
  return inst;
}

void Stats::record(const Event *ev) {
  event_times_.emplace_back(ev->clone());
}

Stats::~Stats(){
  std::ofstream os {"time_stats.json"};
  os  << "{ \"events\" : [\n" ;
  float total = 0;
  for(int i = 0; i < event_times_.size(); i++){
    const auto &evp = event_times_[i];
    evp->print(os);
    if(i < event_times_.size() -1) os << ",\n";
    total += evp->count_ms();
  }
  os << "]}\n";
}
}