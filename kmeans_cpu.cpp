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
  kmeans_srand(Args::s);

  if(Args::i == "tmp.txt") return 0;

  auto d = parseInput(Args::i);
  Centroids cnt = Args::r ? randomCentroids(Args::k, Args::d) 
                          : d.randomCentroidsExcl(Args::k);
  dbg << "########### Initial Centroids\n";
  print_centroids(dbg, cnt);

  {
    DebugStream init_cent {"initial.txt"};
    print_centroids(init_cent, cnt);
  }

  LongCentroids lcnt;
  for(const auto &i: cnt) lcnt.emplace_back(i);

  Problem problem {d, lcnt, Args::m};
  const auto labels = problem.solve();
  dbg << "########### Final Centroids\n";
  print_centroids(dbg, problem.result());
  if(Args::c) print_centroids(cout, problem.result());
  else for(const auto i: labels) cout << i << '\n';

  return 0;
}