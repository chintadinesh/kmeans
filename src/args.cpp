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

}