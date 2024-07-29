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

  Data d {Args::i};
  dbg << "Data = \n";
  for(size_t i = 0; i < d.size(); ++i) dbg << d[i] << '\n';

  DoubleCentroids cnt = Args::r ? randomCentroids(Args::k, Args::d) 
                                : d.randomCentroids(Args::k);
  dbg << "########### Initial Centroids\n";
  print_centroids(dbg, cnt);

  {
    DebugStream init_cent {"initial.txt"};
    print_centroids(init_cent, cnt);
  }

  Kmeans kmeans {d, cnt, Args::m};
  const auto labels = kmeans.fit();
  dbg << "########### Final Centroids\n";
  print_centroids(dbg, kmeans.result());
  if(Args::c) print_centroids(cout, kmeans.result());
  else for(const auto i: labels) cout << i << '\n';

  return 0;
}