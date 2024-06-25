#include "path.h"

using namespace HybridAStar;


//###################################################
//                                         CLEAR PATH
//###################################################

void Path::clear() {
  Node3D node;
  path.poses.clear();
  pathNodes.markers.clear();
  pathVehicles.markers.clear();
  addNode(node, 0);
  addVehicle(node, 1);
  // publishPath();
  // publishPathVel();
  // publishPathNodes();
  // publishPathVehicles();
}

////###################################################
////                                         TRACE PATH
////###################################################
//// __________
//// TRACE PATH
//void Path::tracePath(const Node3D* node, int i) {
//  if (i == 0) {
//    path.header.stamp = ros::Time::now();
//  }

//  if (node == nullptr) { return; }

//  addSegment(node);
//  addNode(node, i);
//  i++;
//  addVehicle(node, i);
//  i++;

//  tracePath(node->getPred(), i);
//}

//###################################################
//                                         TRACE PATH
//###################################################
// __________
// TRACE PATH
void Path::updatePath(const std::vector<const Node3D*>& nodePath) {
  path.header.stamp = ros::Time::now();
  path.header.frame_id = "world";
  int k = 0;
  path.poses.clear();
  pathVel.points.clear();
  for (int i = nodePath.size()-1; i >= 0 ; --i) {
    addSegment(nodePath[i]);
    addNode(*nodePath[i], k);
    k++;
    addVehicle(*nodePath[i], k);
    k++;
  }

  return;
}
// ___________
// ADD SEGMENT
void Path::addSegment(const Node3D* node) {
  geometry_msgs::PoseStamped vertex;
  vertex.pose.position.x = node->getX();
  vertex.pose.position.y = node->getY();
  vertex.pose.position.z = 0;
  vertex.pose.orientation = tf::createQuaternionMsgFromYaw(node->getT());
  // geometry_msgs::Pose vertex;
  // vertex.position.x = node.getX();
  // vertex.position.y = node.getY();
  // vertex.position.z = 0;
  // vertex.orientation = tf::createQuaternionMsgFromYaw(node.getT());
  path.poses.push_back(vertex);
  if (Constants::use_aws){
    const NodeAWS* aws_node = static_cast<const NodeAWS*>(node);
    // const NodeAWS* aws_node = &node;
    trajectory_msgs::JointTrajectoryPoint vertexVel;
    vertexVel.positions = 
      {aws_node->getX(),aws_node->getY(),aws_node->getT()};
    vertexVel.velocities = 
      {aws_node->getVX(),aws_node->getVY(),aws_node->getVYaw()};
    vertexVel.time_from_start = ros::Duration(aws_node->getG());

    pathVel.points.push_back(vertexVel);
  }
}

// ________
// ADD NODE
void Path::addNode(const Node3D& node, int i) {
  visualization_msgs::Marker pathNode;

  // delete all previous markers
  if (i == 0) {
    pathNode.action = 3;
  }

  pathNode.header.frame_id = "world";
  pathNode.header.stamp = ros::Time(0);
  pathNode.id = i;
  pathNode.type = visualization_msgs::Marker::SPHERE;
  pathNode.scale.x = 0.1;
  pathNode.scale.y = 0.1;
  pathNode.scale.z = 0.1;
  pathNode.color.a = 1.0;

  if (smoothed) {
    pathNode.color.r = Constants::pink.red;
    pathNode.color.g = Constants::pink.green;
    pathNode.color.b = Constants::pink.blue;
  } else {
    pathNode.color.r = Constants::purple.red;
    pathNode.color.g = Constants::purple.green;
    pathNode.color.b = Constants::purple.blue;
  }

  pathNode.pose.position.x = node.getX() * Constants::cellSize;
  pathNode.pose.position.y = node.getY() * Constants::cellSize;
  pathNode.pose.orientation = tf::createQuaternionMsgFromYaw(node.getT());
  pathNodes.markers.push_back(pathNode);
}

void Path::addVehicle(const Node3D& node, int i) {
  visualization_msgs::Marker pathVehicle;

  // delete all previous markersg
  if (i == 1) {
    pathVehicle.action = 3;
  }

  pathVehicle.header.frame_id = "world";
  pathVehicle.header.stamp = ros::Time(0);
  pathVehicle.id = i;
  pathVehicle.type = visualization_msgs::Marker::CUBE;
  pathVehicle.scale.x = Constants::length - Constants::bloating * 2;
  pathVehicle.scale.y = Constants::width - Constants::bloating * 2;
  pathVehicle.scale.z = 1;
  pathVehicle.color.a = 0.1;

  if (smoothed) {
    pathVehicle.color.r = Constants::orange.red;
    pathVehicle.color.g = Constants::orange.green;
    pathVehicle.color.b = Constants::orange.blue;
  } else {
    pathVehicle.color.r = Constants::teal.red;
    pathVehicle.color.g = Constants::teal.green;
    pathVehicle.color.b = Constants::teal.blue;
  }

  pathVehicle.pose.position.x = node.getX() * Constants::cellSize;
  pathVehicle.pose.position.y = node.getY() * Constants::cellSize;
  pathVehicle.pose.orientation = tf::createQuaternionMsgFromYaw(node.getT());
  pathVehicles.markers.push_back(pathVehicle);
}
