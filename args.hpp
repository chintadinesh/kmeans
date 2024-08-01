#pragma once
#include <string>
#include <iostream>

namespace utils {
struct Args {
  static unsigned k ; // -k num_cluster: an integer specifying the number of clusters
  static unsigned d ; // -d dims: an integer specifying the dimension of the points
  static std::string  i; // -i inputfilename: a string specifying the input filename
  static unsigned m; // -m max_num_iter: an integer specifying the maximum number of iterations
  static long double t; // -t threshold: a double specifying the threshold for convergence test.
  static bool c; // -c: a flag to control the output of your program. 
                // If -c is specified, your program should output the centroids of all clusters. 
                // If -c is not specified, your program should output the labels of all points. 
  static unsigned s; // -s seed: an integer specifying the seed for rand(). 
                // This is used by the autograder to simplify the correctness checking process. 
  static bool r; // -r randomly choose centroids from 0 to 1. Avoid choosing 
                // centroids from within the points. This is to avoid local minima.
  static bool gpu; // --gpu Run the gpu algirithm. 
  static bool help;

  static void parse_args(int argc, char* argv[]);
  friend std::ostream & operator<<(std::ostream &os, const Args &args); 
};
}