#ifndef ROKAE_DRIVER_SERVOL_MATH_HPP_
#define ROKAE_DRIVER_SERVOL_MATH_HPP_
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include "geometry_msgs/msg/pose.hpp"
namespace rokae_driver::servol {
using Matrix = std::array<double, 16>;
inline constexpr double kElbowSpeedDegS = 5.0;
inline double nextElbowCommand(double current, double joystick_y, double period) {
  if (std::abs(joystick_y) <= 0.15) return current;
  constexpr double pi = 3.14159265358979323846;
  const double velocity = std::copysign(kElbowSpeedDegS * pi / 180.0, joystick_y);
  return std::clamp(current + velocity * period, -pi, pi);
}
inline Matrix matrix(const geometry_msgs::msg::Pose &p) {
  double x=p.orientation.x, y=p.orientation.y, z=p.orientation.z, w=p.orientation.w;
  for (double v : {x,y,z,w,p.position.x,p.position.y,p.position.z})
    if (!std::isfinite(v)) throw std::invalid_argument("pose contains NaN/Inf");
  const double norm=std::hypot(std::hypot(x,y),std::hypot(z,w));
  if (!std::isfinite(norm) || std::abs(norm-1.0)>0.01)
    throw std::invalid_argument("quaternion must be unit length (within 1%)");
  x/=norm; y/=norm; z/=norm; w/=norm;
  return {1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),p.position.x,
          2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),p.position.y,
          2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y),p.position.z,0,0,0,1};
}
inline geometry_msgs::msg::Pose pose(const Matrix &m) {
  for (double value : m)
    if (!std::isfinite(value)) throw std::invalid_argument("matrix contains NaN/Inf");
  geometry_msgs::msg::Pose p;
  p.position.x=m[3]; p.position.y=m[7]; p.position.z=m[11];

  // Convert the row-major rotation matrix to a quaternion.  Choosing the
  // largest diagonal branch avoids loss of precision near 180 degrees.
  const double trace=m[0]+m[5]+m[10];
  double x=0, y=0, z=0, w=0;
  if (trace>0) {
    const double s=2*std::sqrt(std::max(0.0,trace+1));
    w=0.25*s; x=(m[9]-m[6])/s; y=(m[2]-m[8])/s; z=(m[4]-m[1])/s;
  } else if (m[0]>m[5] && m[0]>m[10]) {
    const double s=2*std::sqrt(std::max(0.0,1+m[0]-m[5]-m[10]));
    w=(m[9]-m[6])/s; x=0.25*s; y=(m[1]+m[4])/s; z=(m[2]+m[8])/s;
  } else if (m[5]>m[10]) {
    const double s=2*std::sqrt(std::max(0.0,1+m[5]-m[0]-m[10]));
    w=(m[2]-m[8])/s; x=(m[1]+m[4])/s; y=0.25*s; z=(m[6]+m[9])/s;
  } else {
    const double s=2*std::sqrt(std::max(0.0,1+m[10]-m[0]-m[5]));
    w=(m[4]-m[1])/s; x=(m[2]+m[8])/s; y=(m[6]+m[9])/s; z=0.25*s;
  }
  const double norm=std::hypot(std::hypot(x,y),std::hypot(z,w));
  if (!std::isfinite(norm) || norm<1e-12)
    throw std::invalid_argument("matrix does not contain a valid rotation");
  p.orientation.x=x/norm; p.orientation.y=y/norm;
  p.orientation.z=z/norm; p.orientation.w=w/norm;
  return p;
}
inline Matrix interpolate(const Matrix &a,const Matrix &b,double ratio) {
  const double t=std::clamp(ratio,0.0,1.0);
  const auto pa=pose(a);
  auto pb=pose(b);
  geometry_msgs::msg::Pose result;
  result.position.x=pa.position.x+t*(pb.position.x-pa.position.x);
  result.position.y=pa.position.y+t*(pb.position.y-pa.position.y);
  result.position.z=pa.position.z+t*(pb.position.z-pa.position.z);

  const double ax=pa.orientation.x, ay=pa.orientation.y;
  const double az=pa.orientation.z, aw=pa.orientation.w;
  double bx=pb.orientation.x, by=pb.orientation.y;
  double bz=pb.orientation.z, bw=pb.orientation.w;
  double dot=ax*bx+ay*by+az*bz+aw*bw;
  if (dot<0) { dot=-dot; bx=-bx; by=-by; bz=-bz; bw=-bw; }
  dot=std::clamp(dot,-1.0,1.0);
  double first=1-t, second=t;
  if (dot<0.9995) {
    const double angle=std::acos(dot);
    const double denominator=std::sin(angle);
    first=std::sin((1-t)*angle)/denominator;
    second=std::sin(t*angle)/denominator;
  }
  result.orientation.x=first*ax+second*bx;
  result.orientation.y=first*ay+second*by;
  result.orientation.z=first*az+second*bz;
  result.orientation.w=first*aw+second*bw;
  const double norm=std::hypot(
    std::hypot(result.orientation.x,result.orientation.y),
    std::hypot(result.orientation.z,result.orientation.w));
  result.orientation.x/=norm; result.orientation.y/=norm;
  result.orientation.z/=norm; result.orientation.w/=norm;
  return matrix(result);
}
inline bool withinStep(const Matrix &a,const Matrix &b,double translation,double rotation) {
  for (double v:a) if (!std::isfinite(v)) return false;
  for (double v:b) if (!std::isfinite(v)) return false;
  const double distance=std::hypot(std::hypot(a[3]-b[3],a[7]-b[7]),a[11]-b[11]);
  double trace=0;
  for (int i: {0,1,2,4,5,6,8,9,10}) trace+=a[i]*b[i];
  const double angle=std::acos(std::clamp((trace-1)/2,-1.0,1.0));
  return distance<=translation+1e-12 && angle<=rotation+1e-12;
}
}
#endif
