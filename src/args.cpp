#include "../inc/args.hpp"

using namespace std;

namespace utils {

int Args::k = 0;
int Args::d = 1;
std::string  Args::i = "tmp.txt";
int Args::m = 150; 
double Args::t = 1e-6;
bool Args::c = false;
int Args::s = 1;
bool Args::help = false;

std::ostream & operator<<(std::ostream &os, const Args &args){
  os << "args: \n"
        << " -k : " << args.k << '\n'
        << " -d : " << args.d << '\n'
        << " -i : " << args.i << '\n'
        << " -m : " << args.m << '\n'
        << " -t : " << args.t << '\n'
        << " -c : " << args.c << '\n'
        << " -s : " << args.s << '\n';
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
            << " -c: a flag to control the output of your program.\n" 
            << "      If -c is specified, your program should output the centroids of all clusters. \n"
            << "      If -c is not specified, your program should output the labels of all points.\n"
            << " -s seed: an integer specifying the seed for rand(). \n"
            << "      This is used by the autograder to simplify the correctness checking process.\n";
      help = true;
      return;
    } else if ((arg == "-k") && j + 1 < argc) {
      k = std::stoi(argv[++j]);
    } else if ((arg == "-d") && j + 1 < argc) {
      d = std::stoi(argv[++j]);
    } else if ((arg == "-i") && j + 1 < argc) {
      i = string(argv[++j]);
    } else if ((arg == "-m") && j + 1 < argc) {
      m = std::stoi(argv[++j]);
    } else if ((arg == "-t") && j + 1 < argc) {
      t = std::stoi(argv[++j]);
    } else if ((arg == "-c") && j + 1 < argc) {
      c = std::stoi(argv[++j]);
    } else if ((arg == "-s") && j + 1 < argc) {
      s = std::stoi(argv[++j]);
    } 
    else {
        std::cerr << "Unknown option: " << arg << "\n";
        return;
    }
  }
}

}