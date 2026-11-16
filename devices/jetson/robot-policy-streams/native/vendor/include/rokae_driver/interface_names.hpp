#ifndef ROKAE_DRIVER__INTERFACE_NAMES_HPP_
#define ROKAE_DRIVER__INTERFACE_NAMES_HPP_

#include <string>

namespace rokae_driver::interface_names {

// Nodes and business interfaces share the delivered upper-limb namespace.
inline constexpr char kNodeNamespace[] = "/robot_02/upperlimb";
inline constexpr char kUpperlimbPrefix[] = "/robot_02/upperlimb";
inline constexpr char kInitializeDualArm[] =
    "/robot_02/upperlimb/initialize/dual_arm";

inline std::string arm(const std::string &function,
                       const std::string &side) {
  return std::string(kUpperlimbPrefix) + "/" + function + "/" +
         side + "_arm";
}

inline std::string dualArm(const std::string &function) {
  return std::string(kUpperlimbPrefix) + "/" + function + "/dual_arm";
}

inline std::string nodeNamespace(const std::string &side) {
  return std::string(kNodeNamespace) + "/" + side;
}

}  // namespace rokae_driver::interface_names

#endif  // ROKAE_DRIVER__INTERFACE_NAMES_HPP_
