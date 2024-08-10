#include "args.hpp"
#include <filesystem>

using namespace std;

namespace kmeans {

unsigned Args::k = 0;
unsigned Args::d = 0;
std::string  Args::i = "tmp.txt";
unsigned Args::m = 150; 
long double Args::t = 1e-12;
bool Args::c = false;
bool Args::r = false;
unsigned Args::s = 1;
bool Args::help = false;
std::string Args::gpu = "";
unsigned Args::threads_classify = 128;
unsigned Args::blocks_classify = 0;
unsigned Args::threads_update = 128;
unsigned Args::blocks_update = 0;

unsigned Args::data_size = 0;

bool Args::debug = false;

std::ostream & operator<<(std::ostream &os, const Args &args){
  os << "args: \n"
        << " -k : " << args.k << '\n'
        << " -d : " << args.d << '\n'
        << " -i : " << args.i << '\n'
        << " -m : " << args.m << '\n'
        << " -t : " << args.t << '\n'
        << " -c : " << args.c << '\n'
        << " -s : " << args.s << '\n'
        << " -r: " << args.r << '\n'
        << " --gpu: " << args.gpu << '\n'
        << " --threads_classify: " << args.threads_classify << '\n'
        << " --threads_update: " << args.threads_update << '\n'
        << " --gpu: " << args.gpu << '\n'
        << " --debug: " << args.debug << '\n';
  return os;
}

void Args::parse_args(int argc, char* argv[]){
  for(int j = 1; j < argc; ++j){
    std::string arg = argv[j];
    if (arg == "-h" || arg == "--help") {
      cout << "Usage: " << argv[0] << " options\n"
            << " -k num_cluster: an integer specifying the number of clusters.\n"
            << " -d dims: an integer specifying the dimension of the points.\n"
            << " -i inputfilename: a string specifying the input filename.\n"
            << " -m max_num_iter: an integer specifying the maximum number of iterations.\n"
            << " -t threshold: a double specifying the threshold for convergence test.\n"
            << " -c a flag to control the output of your program.\n" 
            << "      If -c is specified, your program should output the centroids of all clusters. \n"
            << "      If -c is not specified, your program should output the labels of all points.\n"
            << " -s seed: an integer specifying the seed for rand(). \n"
            << "      This is used by the autograder to simplify the correctness checking process.\n"
            << " -r randomly choose centroids from 0 to 1. Avoid choosing\n" 
            << "      centroids from within the points. This is to avoid local minima.\n"
            << " --gpu [baseline]\n"
            << "       Run Kmeans on gpu with the specified strategy.\n"
            << " --threads_classify. Number of threads per block to perform\n"
            << "      classification. #Blocks are computed accordingly."
            << " --threads_update. Number of threads per block to perform\n"
            << "      centroid updation. #Blocks are computed accordingly."
            << " --debug Redirect debug info to cerr."; 
      help = true;
      return;
    } else if ((arg == "-k") && j + 1 < argc) {
      k = stoi(argv[++j]);
    } else if ((arg == "-d") && j + 1 < argc) {
      d = stoi(argv[++j]);
    } else if ((arg == "-i") && j + 1 < argc) {
      i = string(argv[++j]);
    } else if ((arg == "-m") && j + 1 < argc) {
      m = stoi(argv[++j]);
    } else if ((arg == "-t") && j + 1 < argc) {
      t = stod(argv[++j]);
    } else if (arg == "-c") {
      c = true;
    } else if (arg == "-r") {
      r = true;
    } else if ((arg == "--gpu") && j + 1 < argc) {
      gpu = argv[j+1];
      if(gpu != "" || gpu != "baseline"){
        cerr << "ERROR: invalid gpu option: " << gpu << '\n';
        cerr << "       Valid options are baseline\n";
        exit(EXIT_FAILURE);
      }
    } else if ((arg == "--threads_classify") && j + 1 < argc) {
      threads_classify = stoi(argv[++j]);
      if((threads_classify & (threads_classify-1)) != 0){
        cerr << "ERROR: threads_classify should be power of 2.\n";
        exit(EXIT_FAILURE);
      }
    } else if ((arg == "--threads_update") && j + 1 < argc) {
      threads_update = stoi(argv[++j]);
      if((threads_update & (threads_update-1)) != 0){
        cerr << "ERROR: threads_update should be power of 2.\n";
        exit(EXIT_FAILURE);
      }
    } else if (arg == "--debug") {
      debug = true;
    } else if ((arg == "-s") && j + 1 < argc) {
      s = stoi(argv[++j]);
    } 
    else {
        cerr << "Unknown option: " << arg << "\n";
        return;
    }
  }

  if(k<=0) cerr << "Invalid clusters: " << k << '\n';
  if(d<=0) cerr << "Invalid dimensions: " << d << '\n';
  if(!filesystem::exists(i)) cerr << "Input file does not exist: " << i << '\n';
}

void Args::set_data_size(const unsigned sz){
  if(!sz){
    cerr << "ERROR: Invalid data size. Should be > 0.\n";
    exit(EXIT_FAILURE);
  }

  data_size = sz;

  // compute the block dimensions of GPU kernels
  blocks_classify = (sz + threads_classify - 1) / threads_classify;
  blocks_update = (sz + threads_update - 1) / threads_update;
}

}