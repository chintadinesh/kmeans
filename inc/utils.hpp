#pragma once

#include <vector>
#include <iostream>

namespace utils {
  using Point = std::vector<double>;
  using Data = std::vector<Point>;

  Data parseInput(const std::string &input_file);
  std::ostream & operator<<(std::ostream &os, const Data &data);
}