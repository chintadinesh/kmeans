#include "args.hpp"
#include "utils.hpp"

using namespace kmeans;
using namespace std;

extern DebugStream dbg;

int main(int argc, char* argv[]){
  Args::parse_args(argc, argv);
  if(Args::help)
    return 0;
  
  if(Args::debug) dbg.setOutputStream(cerr);

  dbg << Args(); 
  kmeans_srand(Args::s);

  if(Args::i == "tmp.txt") return 0;

  Data d {Args::i};
  //dbg << "Data = \n";
  //for(size_t i = 0; i < d.size(); ++i) dbg << d[i] << '\n';
  dbg << "Read data successfully\n";

  std::unique_ptr<Kmeans> km {kmeansFactory(d, Args::r, Args::k, Args::m, Args::gpu)};

  dbg << "Kmeans created\n";

  const auto labels = km->fit();

  dbg << "########### Final Centroids\n";
  print_centroids(dbg, km->result());

  if(Args::c) print_centroids(cout, km->result());
  else for(size_t i = 0; i < labels.size(); ++i) cout << labels[i] << '\n';

  return 0;
}