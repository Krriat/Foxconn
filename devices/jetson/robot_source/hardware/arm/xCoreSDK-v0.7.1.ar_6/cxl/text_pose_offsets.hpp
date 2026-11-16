#pragma once
#include <array>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <string>

namespace text_points {
struct Offset {
  std::string base;
  std::array<double, 3> xyz{};
};
inline Offset flattenOffset(const std::string &name, const std::map<std::string, Offset> &offsets) {
  Offset result{name, {0,0,0}};
  std::set<std::string> visited;
  while (offsets.count(result.base)) {
    if (!visited.insert(result.base).second) throw std::runtime_error("Offset point cycle: " + name);
    const auto &item = offsets.at(result.base);
    for (int i=0;i<3;++i) {
      result.xyz[i] += item.xyz[i];
      if (!std::isfinite(result.xyz[i])) throw std::runtime_error("Non-finite offset: " + name);
    }
    result.base = item.base;
  }
  return result;
}
inline std::array<double,6> applyOffset(std::array<double,6> pose, const std::array<double,3> &offset) {
  for (int i=0;i<3;++i) pose[i] += offset[i];
  for (double v:pose) if (!std::isfinite(v)) throw std::runtime_error("Non-finite derived pose");
  return pose;
}
}
