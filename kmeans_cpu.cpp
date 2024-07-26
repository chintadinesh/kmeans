#include "args.hpp"
#include "utils.hpp"

using namespace utils;
using namespace std;

DebugStream dbg{"debug.log"};

int main(int argc, char* argv[]){
  Args::parse_args(argc, argv);
  if(Args::help)
    return 0;

  dbg << utils::Args(); 

  if(Args::i == "tmp.txt") return 0;

  auto d = parseInput(Args::i);
  auto cnt = d.randomCentroids(Args::k);
  dbg << "########### Initial Centroids\n";
  print_centroids(dbg, cnt);

  LongCentroids lcnt;
  for(const auto &i: cnt) lcnt.emplace_back(i);

  Problem problem {d, lcnt, Args::m};
  const auto labels = problem.solve();
  //cout << "########### Final Centroids\n";
  //print_centroids(final_cnt);
  if(Args::c) print_centroids(dbg, problem.result());
  else for(const auto i: labels) cout << i << '\n';

  return 0;
}