#include "../inc/args.hpp"
#include "../inc/utils.hpp"

using namespace utils;
using namespace std;

int main(int argc, char* argv[]){
  Args::parse_args(argc, argv);
  if(Args::help)
    return 0;

  cout << utils::Args(); 

  if(Args::i != "tmp.txt"){
    auto d = parseInput(Args::i);
    cout << d;
  }

  return 0;
}