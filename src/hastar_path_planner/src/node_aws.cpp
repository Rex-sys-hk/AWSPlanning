#include "node_aws.h"
#include <algorithm>
#include <iostream>

using namespace HybridAStar;

// CONSTANT VALUES
// possible directions
// const int NodeAWS::dir = 3;
// possible movements
//const float NodeAWS::dy[] = { 0,        -0.032869,  0.032869};
//const float NodeAWS::dx[] = { 0.62832,   0.62717,   0.62717};
//const float NodeAWS::dt[] = { 0,         0.10472,   -0.10472};

// R = 6, 6.75 DEG

// std::vector<float> range_sample(int n, float lb, float hb) {
//   std::vector<float> result = {};
//   float step = (hb - lb) / n;
//   for (int i = 0; i < n; ++i) {
//     result.emplace_back(lb + i * step);
//   }
//   return result;
// }
// PI/6

// // const float NodeAWS::aws_dy[] = {0.,  1./2,   0.3535533905932738,  0., -0.3535533905932738, -1./2, 0.3535533905932738,  0.,  0.3535533905932738};
// const float NodeAWS::aws_dy[] = {0.,  1.,  1,  0.7068582,  0., -0.7068582, -1., -1, -0.7068582,  0.,  0.7068582};
// // const float NodeAWS::aws_dx[] = {0.,  0.,   0.3535533905932738,  1./2,  0.3535533905932738,  0.,  -0.3535533905932738, -1./2, -0.3535533905932738};
// const float NodeAWS::aws_dx[] = {0.,  0.01,  -0.01, 0.7068582,  1.,  0.7068582,  0.01, -0.01, -0.7068582, -1., -0.7068582};

// const float NodeAWS::aws_dt[] = {0, M_PI/9, -M_PI/9};
// const std::vector<float> NodeAWS::s_theta = range_sample(21, -M_PI, M_PI);
// const std::vector<float> NodeAWS::s_r = range_sample(11, -1.0 / Constants::r, 1.0 / Constants::r);
// const std::vector<float> NodeAWS::s_k = range_sample(11, -1.0 / Constants::r, 1.0 / Constants::r);

// const std::vector<float> NodeAWS::s_dyaw = range_sample(5, -M_PI_2, M_PI_2);
// const std::vector<float> NodeAWS::s_ds = range_sample(5, -5, 5);


// R = 3, 6.75 DEG
//const float NodeAWS::dy[] = { 0,        -0.0207946, 0.0207946};
//const float NodeAWS::dx[] = { 0.35342917352,   0.352612,  0.352612};
//const float NodeAWS::dt[] = { 0,         0.11780972451,   -0.11780972451};

//const float NodeAWS::dy[] = { 0,       -0.16578, 0.16578};
//const float NodeAWS::dx[] = { 1.41372, 1.40067, 1.40067};
//const float NodeAWS::dt[] = { 0,       0.2356194,   -0.2356194};

bool NodeAWS::isInRange(const NodeAWS& goal) const {
  return std::abs(getX() - goal.getX()) < Constants::cellSize &&
         std::abs(getY() - goal.getY()) < Constants::cellSize &&
         std::abs(Helper::normalizeHeadingRad(getT() - goal.getT())) < Constants::deltaHeadingRad;
}

//###################################################
//                                   CREATE SUCCESSOR
//###################################################
NodeAWS* NodeAWS::createSuccessor(const int i) {
  float xSucc;
  float ySucc;
  float tSucc;
  // forward_sim
  // calculate successor positions forward
  // if (i < 4) {
  std::vector<float> vi = sampling_table->at(i);
  xSucc = getX() + vi[0] * cos(getT()) - vi[1] * sin(getT());
  ySucc = getY() + vi[0] * sin(getT()) + vi[1] * cos(getT());
  tSucc = Helper::normalizeHeadingRad(getT() + vi[2]);
  // }
  // // backwards
  // else {
  //   xSucc = getX() - dx[i - 3] * cos(getT()) - dy[i - 3] * sin(getT());
  //   ySucc = getY() - dx[i - 3] * sin(getT()) + dy[i - 3] * cos(getT());
  //   tSucc = Helper::normalizeHeadingRad(getT() - dt[i - 3]);
  // }
  // std::cout<<"aws_dx: "<<aws_dx[(int)floor(i/3)]<<" aws_dy: "<<aws_dy[(int)floor(i/3)]<<" dt: "<<dt[(int)i%3]<<std::endl;
  return new NodeAWS(xSucc, ySucc, tSucc, getG(), 0., 
                    vi[3], vi[4], vi[5], i, 
                    this, sampling_table);
}


// 定义车辆的角速度和角速度控制器
NodeAWS::WheelCtrlCmd NodeAWS::RelativeVAndOmegaControl(double vx, double vy, double omega, double lim_angle, bool wheel_rotate_velocity) {
    // vx, vy, omega are the robot's velocity in robot frame
    // v is each wheel speed
    // s is each steering angle
    // half_l = VehicleConst.WHEEL_BASE/2
    // half_w = VehicleConst.WHEEL_WHIDTH/2
    // double wheel_r = Constants::WHEEL_RADIUS;

    // robot heading is x axis, wheel sequence FL, FR, RL, RR
    // wheel_positions = np.array([[half_l,half_w],[half_l,-half_w],[-half_l,half_w],[-half_l,-half_w]])
    auto wheel_positions_x=Constants::wheel_positions_x;// = {half_l, half_l, -half_l, -half_l}
    // wheel_positions_x<<half_l, half_l, -half_l, -half_l;
    auto wheel_positions_y=Constants::wheel_positions_y;// = {half_w, -half_w, -half_w, half_w};
    // wheel_positions_y<<half_w, -half_w, -half_w, half_w;
    WheelCtrlCmd wheel_ctrl_cmd;

    // 计算每个轮子的角速度
    for (int i = 0; i < (int)wheel_positions_x.size(); ++i) {
      auto wheel_vx = vx - wheel_positions_y.at(i) * omega;
      auto wheel_vy = vy + wheel_positions_x.at(i) * omega;
      double wheel_v = sqrt(wheel_vx * wheel_vx + wheel_vy * wheel_vy);
      double wheel_s = atan2(wheel_vy, wheel_vx);

      // 如果角速度小于限制角度，则将负号加入
      if (Constants::has_steer_limit){
      float limit_angle_u = Constants::wheel_steer_limits_up[i];
      float limit_angle_l = Constants::wheel_steer_limits_low[i];
      bool oob_wheel_ub = (wheel_s > limit_angle_u) 
                       && (Helper::normalizeHeadingRad(wheel_s+M_PI) 
                       > limit_angle_l);
      bool oob_wheel_lb = (wheel_s < limit_angle_l) 
                       && (Helper::normalizeHeadingRad(wheel_s-M_PI)
                       < limit_angle_u);
      bool oob_wheel = oob_wheel_ub || oob_wheel_lb;
      if (oob_wheel) {
        wheel_s = Helper::normalizeHeadingRad(wheel_s + M_PI);
        wheel_v = -wheel_v;
      }
      }

      wheel_ctrl_cmd.first.push_back(wheel_v);
      wheel_ctrl_cmd.second.push_back(wheel_s);
    }

    return wheel_ctrl_cmd;
}
//###################################################
//                                      MOVEMENT COST
//###################################################
void NodeAWS::updateG() {
  //calculate wheel pose_diff/speed 
  double wholebody_v_t = sqrt(pow(getVX(),2)+pow(getVY(),2)) \
                         / Constants::max_wheel_v;
  double wholebody_w_t = fabs(getVYaw()/Constants::max_steering_v);
  double wb_cost = std::max(wholebody_v_t, wholebody_w_t);
  if (getPred() == nullptr) {
    getG_mutable() += wb_cost;
    return;
  }
  auto pred_ctrl = RelativeVAndOmegaControl(
    getPred()->getVX(), getPred()->getVY(), getPred()->getVYaw());
  auto succ_ctrl = RelativeVAndOmegaControl(
    getVX(), getVY(), getVYaw());
  std::vector<double> diff_wheel_speed;
  std::vector<double> diff_wheel_pose;
  std::vector<double> maneuver_time;

  for (int i = 0; i < (int)pred_ctrl.first.size(); ++i) {
    diff_wheel_speed.push_back( 
      std::abs(- pred_ctrl.first.at(i) + succ_ctrl.first.at(i))
      /Constants::max_wheel_a
      );
    diff_wheel_pose.push_back( 
      std::abs(- pred_ctrl.second.at(i) + succ_ctrl.second.at(i))
      /Constants::max_steering_v
      );
    maneuver_time.push_back(
      pow(
        pow(diff_wheel_speed.at(i),2) + pow(diff_wheel_pose.at(i),2),
        0.5)
      );
  }

  double max_wheel_pose_diff = *std::max_element(
                diff_wheel_pose.begin(), diff_wheel_pose.end());
  double max_wheel_speed_diff = *std::max_element(
                diff_wheel_speed.begin(), diff_wheel_speed.end());
  double max_wheel_t = *std::max_element(
                maneuver_time.begin(), maneuver_time.end());

  getG_mutable() = getG() + std::max(wb_cost, max_wheel_t);
  // std::max(max_wheel_speed_diff/Constants::max_wheel_a ,max_wheel_pose_diff/Constants::max_steering_v); 

}
