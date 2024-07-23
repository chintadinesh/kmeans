#include "../inc/args.hpp"

using namespace utils;
int main(int argc, char* argv[]){
  Args::parse_args(argc, argv);
  if(Args::help)
    return 0;

  std::cout << utils::Args(); 
  return 0;
}