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

  std::unique_ptr<Kmeans> kmeans {kmeansFactory(d, Args::r, Args::k, Args::m)};

  const auto labels = kmeans->fit();

  dbg << "########### Final CPU Centroids\n";
  print_centroids(dbg, kmeans->result());

  if(Args::c) print_centroids(cout, kmeans->result());
  else for(size_t i = 0; i < labels.size(); ++i) cout << labels[i] << '\n';

  return 0;
}