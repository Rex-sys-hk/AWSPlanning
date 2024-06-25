#ifndef NodeAWS_H
#define NodeAWS_H

#include <cmath>
#include <iostream>
#include <vector>

#include "node3d.h"

namespace HybridAStar {

class NodeAWS : public Node3D{
 public:
  NodeAWS(): NodeAWS(0, 0, 0, 0, 0, 0, 0, 0, 0, nullptr, nullptr) {}
  NodeAWS(float x, float y, float t, float g, float h, 
          float vx, float vy, float vyaw, int prim = 0,
          const NodeAWS* pred=nullptr, 
          const std::vector<std::vector<float>> *sampling_table=nullptr
          )
          :Node3D(x, y, t, g, h, pred, prim) {
    // this->ICM_theta = icm_theta;
    // this->ICM_r = icm_r;
    _vx = vx;
    _vy = vy;
    _vyaw = vyaw;
    this->pred = pred;
    sampling_table = sampling_table;
  }
  NodeAWS(const Node3D& node3d, float vx=0, float vy=0, float vyaw=0, 
          const NodeAWS* pred=nullptr, 
          const std::vector<std::vector<float>> *sampling_table=nullptr)
          :Node3D(node3d) {
    // this->ICM_theta = icm_theta;
    // this->ICM_r = icm_r;
    _vx = vx;
    _vy = vy;
    _vyaw = vyaw;
    this->pred = pred;
    sampling_table = sampling_table;
  }
  ~NodeAWS() {} 

  // const float getVX() const { return _vx; }
  // const float getVY() const { return _vy; }
  // const float getVYaw() const { return _vyaw; }

  // void setVX(const float& vx) { _vx = vx; }
  // void setVY(const float& vy) { _vy = vy; }
  // void setVYaw(const float& vyaw) { _vyaw = vyaw; }
  const NodeAWS* getPred() const { return pred; }
  void setPred(const NodeAWS* pred) { this->pred = pred; }
  
  // float getICM_theta() const { return ICM_theta; }
  // float getICM_r() const { return ICM_r; }
  // float getDYaw() const { return ICM_d_yaw; }
  // float getDS() const { return ICM_d_s; }

  // float setICM_x(const float& icm_theta) { this->ICM_theta = icm_theta; }
  // float setICM_y(const float& icm_r) { this->ICM_r = icm_r; }
  // float setOmega(const float& icm_d_yaw) { this->ICM_d_yaw = icm_d_yaw; }
  // float setDS(const float& icm_d_s) { this->ICM_d_s = icm_d_s; }
  bool isInRange(const NodeAWS& goal) const;
  NodeAWS* createSuccessor(const int i);
  typedef std::pair<std::vector<double>,std::vector<double>> WheelCtrlCmd;
  WheelCtrlCmd RelativeVAndOmegaControl(double vx, double vy, double omega, 
                                        double lim_angle = M_PI_2, 
                                        bool wheel_rotate_velocity = false);

  void updateG();

  // bool NodeAWS::operator == (const NodeAWS& rhs) const;

  // static const std::vector<float> s_theta;
  // static const std::vector<float> s_r;
  // static const std::vector<float> s_k;
  // static const std::vector<float> s_dyaw;
  // static const std::vector<float> s_ds;
  /// Possible movements in the x direction
  static const float aws_dx[];
  /// Possible movements in the y direction
  static const float aws_dy[];
  /// Possible movements regarding heading theta
  static const float aws_dt[];

  const std::vector<std::vector<float>> *sampling_table;

 private:
    // instantanious center of motion
    // float _vx;
    // float _vy;
    // float _vyaw;
    const NodeAWS* pred;
    // float ICM_theta; // polar coord theta, [-PI,PI]
    // float ICM_r; // polar coord radius, [-inf,+inf)
    // float ICM_k; // curvature, dependent on ICM_r, max 1/r
    // float ICM_d_yaw; // yaw change, [-PI/2,PI/2], dependent on ICM_r, ICM_d_s, max PI/2
    // float ICM_d_s; // curve length, [-5, 5], dependent on ICM_r and ICM_d_yaw, max 5m
    // float manip_cost; // online calculation
    // float manip_heur; // TODO
};

} // namespace HybridAStar
#endif // NodeAWS_H