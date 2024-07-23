#pragma once
#include <string>
#include <iostream>

namespace utils {
struct Args {
  static int k ; // -k num_cluster: an integer specifying the number of clusters
  static int d ; // -d dims: an integer specifying the dimension of the points
  static std::string  i; // -i inputfilename: a string specifying the input filename
  static int m; // -m max_num_iter: an integer specifying the maximum number of iterations
  static double t; // -t threshold: a double specifying the threshold for convergence test.
  static bool c; // -c: a flag to control the output of your program. 
                // If -c is specified, your program should output the centroids of all clusters. 
                // If -c is not specified, your program should output the labels of all points. 
  static int s; // -s seed: an integer specifying the seed for rand(). 
                // This is used by the autograder to simplify the correctness checking process. 

  friend std::ostream & operator<<(std::ostream &os, const Args &args); 
};
}