#include "algorithm.h"

#include <boost/heap/binomial_heap.hpp>
#include <Eigen/Dense>

using namespace HybridAStar;

float aStar(Node2D& start, Node2D& goal, Node2D* nodes2D, int width, int height, CollisionDetection& configurationSpace, Visualize& visualization);
void updateH(Node3D& start, const Node3D& goal, Node2D* nodes2D, float* dubinsLookup, int width, int height, CollisionDetection& configurationSpace, Visualize& visualization);
Node3D* dubinsShot(Node3D& start, const Node3D& goal, CollisionDetection& configurationSpace);
NodeAWS* dubinsShot(NodeAWS& start, const NodeAWS& goal, CollisionDetection& configurationSpace);
std::vector<std::vector<float>> AWSConstructSamplingTable(int r_samples, int psi_samples, std::vector<float> omega_sample_list = {M_PI_2, M_PI/3, M_PI_4, M_PI/6}, float delta_cur=1.);
template<typename T>
std::vector<float> linspace(T start_in, T end_in, int num_in);
//###################################################
//                                    NODE COMPARISON
//###################################################
/*!
   \brief A structure to sort nodes in a heap structure
*/
struct CompareNodes {
  /// Sorting 3D nodes by increasing C value - the total estimated cost
  bool operator()(const Node3D* lhs, const Node3D* rhs) const {
    return lhs->getC() > rhs->getC();
  }
  /// Sorting 2D nodes by increasing C value - the total estimated cost
  bool operator()(const Node2D* lhs, const Node2D* rhs) const {
    return lhs->getC() > rhs->getC();
  }
};

struct CompareNodesH {
  /// Sorting 3D nodes by increasing C value - the total estimated cost
  bool operator()(const Node3D* lhs, const Node3D* rhs) const {
    return lhs->getH() > rhs->getH();
  }
  /// Sorting 2D nodes by increasing C value - the total estimated cost
  bool operator()(const Node2D* lhs, const Node2D* rhs) const {
    return lhs->getH() > rhs->getH();
  }
};

//###################################################
//                                        3D A*
//###################################################
Node3D* Algorithm::hybridAStar(Node3D& start,
                               const Node3D& goal,
                               Node3D* nodes3D,
                               Node2D* nodes2D,
                               int width,
                               int height,
                               CollisionDetection& configurationSpace,
                               float* dubinsLookup,
                               Visualize& visualization) {

  // PREDECESSOR AND SUCCESSOR INDEX
  int iPred, iSucc;
  float newG;
  // Number of possible directions, 3 for forward driving and an additional 3 for reversing
  int dir = Constants::reverse ? 6 : 3;
  // Number of iterations the algorithm has run for stopping based on Constants::iterations
  int iterations = 0;

  // VISUALIZATION DELAY
  ros::Duration d(0.003);

  // OPEN LIST AS BOOST IMPLEMENTATION
  typedef boost::heap::binomial_heap<Node3D*,
          boost::heap::compare<CompareNodes>
          > priorityQueue;
  priorityQueue O;

  // update h value
  updateH(start, goal, nodes2D, dubinsLookup, width, height, configurationSpace, visualization);
  // mark start as open
  start.open();
  // push on priority queue aka open list
  O.push(&start);
  iPred = start.setIdx(width, height);
  nodes3D[iPred] = start;

  // NODE POINTER
  Node3D* nPred;
  Node3D* nSucc;

  // float max = 0.f;

  // continue until O empty
  while (!O.empty()) {

    //    // DEBUG
    //    Node3D* pre = nullptr;
    //    Node3D* succ = nullptr;

    //    std::cout << "\t--->>>" << std::endl;

    //    for (priorityQueue::ordered_iterator it = O.ordered_begin(); it != O.ordered_end(); ++it) {
    //      succ = (*it);
    //      std::cout << "VAL"
    //                << " | C:" << succ->getC()
    //                << " | x:" << succ->getX()
    //                << " | y:" << succ->getY()
    //                << " | t:" << helper::toDeg(succ->getT())
    //                << " | i:" << succ->getIdx()
    //                << " | O:" << succ->isOpen()
    //                << " | pred:" << succ->getPred()
    //                << std::endl;

    //      if (pre != nullptr) {

    //        if (pre->getC() > succ->getC()) {
    //          std::cout << "PRE"
    //                    << " | C:" << pre->getC()
    //                    << " | x:" << pre->getX()
    //                    << " | y:" << pre->getY()
    //                    << " | t:" << helper::toDeg(pre->getT())
    //                    << " | i:" << pre->getIdx()
    //                    << " | O:" << pre->isOpen()
    //                    << " | pred:" << pre->getPred()
    //                    << std::endl;
    //          std::cout << "SCC"
    //                    << " | C:" << succ->getC()
    //                    << " | x:" << succ->getX()
    //                    << " | y:" << succ->getY()
    //                    << " | t:" << helper::toDeg(succ->getT())
    //                    << " | i:" << succ->getIdx()
    //                    << " | O:" << succ->isOpen()
    //                    << " | pred:" << succ->getPred()
    //                    << std::endl;

    //          if (pre->getC() - succ->getC() > max) {
    //            max = pre->getC() - succ->getC();
    //          }
    //        }
    //      }

    //      pre = succ;
    //    }

    // pop node with lowest cost from priority queue
    nPred = O.top();
    // set index
    iPred = nPred->setIdx(width, height);
    iterations++;

    // RViz visualization
    if (Constants::visualization) {
      visualization.publishNode3DPoses(*nPred);
      visualization.publishNode3DPose(*nPred);
      // d.sleep();
    }

    // _____________________________
    // LAZY DELETION of rewired node
    // if there exists a pointer this node has already been expanded
    if (nodes3D[iPred].isClosed()) {
      // pop node from the open list and start with a fresh node
      O.pop();
      continue;
    }
    // _________________
    // EXPANSION OF NODE
    else if (nodes3D[iPred].isOpen()) {
        // add node to closed list
        nodes3D[iPred].close();
        // remove node from open list
        O.pop();

        // _________
        // GOAL TEST
        if (*nPred == goal || iterations > Constants::iterations) {
          // DEBUG
          if (iterations > Constants::iterations) std::cout << "iterations exceeded" << std::endl;
          else std::cout<<"goal found" <<std::endl;
          
          return nPred;
        }

        // ____________________
        // CONTINUE WITH SEARCH
        // _______________________
        // SEARCH WITH DUBINS SHOT
        std::cout<< "dubinsShot in range: "<< nPred->isInRange(goal)<<" Prim: "<< nPred->getPrim() << std::endl;
        if (Constants::dubinsShot && nPred->isInRange(goal) && nPred->getPrim() < 3) {
          nSucc = dubinsShot(*nPred, goal, configurationSpace);
          std::cout << "dubinsShot: "<< nSucc << std::endl;
          if (nSucc != nullptr && *nSucc == goal) {
            //DEBUG
            // std::cout << "max diff " << max << std::endl;
            return nSucc;
          }
        }

        // ______________________________
        // SEARCH WITH FORWARD SIMULATION
      for (int i = 0; i < dir; i++) {
        // create possible successor
        nSucc = nPred->createSuccessor(i);
        if(nSucc==nPred){
          std::cout << "looping at start"<<std::endl;
          return nSucc;
        }
        // set index of the successor
        iSucc = nSucc->setIdx(width, height);

        // ensure successor is on grid and traversable
        if (!nSucc->isOnGrid(width, height) || !configurationSpace.isTraversable(nSucc)) 
        {
          // std::cout<< "succ is not on grid and traversable" << std::endl;
          delete nSucc;
          continue;
        }
        // calculate new G value
        nSucc->updateG();
        newG = nSucc->getG(); // get updated g value
        // ensure successor is not on closed list or it has the same index as the predecessor
        // if successor not on open list or found a shorter way to the cell
        if (nodes3D[iSucc].isClosed() && iPred != iSucc && newG >= nodes3D[iSucc].getG())
        {
          // std::cout<< "succ closed or not same as Pred or with higher G" << std::endl;
          delete nSucc; 
          continue;
        }
        updateH(*nSucc, goal, nodes2D, dubinsLookup, width, height, configurationSpace, visualization);
        // if(iSucc == iPred && newG <= nodes3D[iSucc].getG() && nPred->getPred() != nullptr){
        //   std::cout << "iSucc == iPred"<<std::endl;
        //   std::cout << nPred->getPred()<<std::endl;
        //   nSucc->setPred(nPred->getPred()); 
        //   nSucc->updateG();
        // }
        if(nSucc==nSucc->getPred()){
          std::cout << "looping after H update"<<std::endl;
          return nSucc;
        }
        if (iSucc == iPred && nSucc->getC() > nPred->getC() - Constants::tieBreaker)// + Constants::tieBreaker) 
        {
          // std::cout<< "succ in same cell with larger C" << std::endl;
          delete nSucc;
          continue;
        }
        if (nSucc->getPred() == nSucc) {
          std::cout << "looping in exploration"<<std::endl;
          return nullptr;
        }
        // put successor on open list
        nSucc->open();
        nodes3D[iSucc] = *nSucc;
        // O.push(&nodes3D[iSucc]);
        O.push(nSucc);
        // delete nSucc;
      }
      }
    }

  if (O.empty()) {
    std::cout << "No candidate state, path not found" << std::endl;
    return nullptr;
  }

  return nullptr;
}


NodeAWS* Algorithm::hybridAStar(NodeAWS& start,
                               const NodeAWS& goal,
                               NodeAWS* nodes3D,
                               Node2D* nodes2D,
                               int width,
                               int height,
                               CollisionDetection& configurationSpace,
                               float* dubinsLookup,
                               Visualize& visualization) {

  // PREDECESSOR AND SUCCESSOR INDEX
  int iPred, iSucc;
  float newG;
  // Number of possible directions, 3 for forward driving and an additional 3 for reversing
  // int dir = Constants::reverse ? 6 : 3;
  // Number of iterations the algorithm has run for stopping based on Constants::iterations
  int iterations = 0;

  // VISUALIZATION DELAY
  ros::Duration d(0.003);

  // OPEN LIST AS BOOST IMPLEMENTATION
  typedef boost::heap::binomial_heap<NodeAWS*,
          boost::heap::compare<CompareNodes>
          > priorityQueue;
  priorityQueue O;
  typedef boost::heap::binomial_heap<NodeAWS*,
          boost::heap::compare<CompareNodesH>
          > priorityQueueH;
  priorityQueueH O_H;
  // init sampling table
  const auto sampling_table = AWSConstructSamplingTable(
                        Constants::r_sample_num,
                        Constants::psi_sample_num, 
                        // {M_PI_2, M_PI/3, M_PI_4, M_PI/6}
                        linspace(-M_PI_4, M_PI_4, 
                                  Constants::omega_sample_num),
                        sqrt(2*Constants::cellSize*Constants::cellSize)/2
                        );
  // int dir = 11*3;
  int dir = sampling_table.size();
  // update h value
  updateH(start, goal, nodes2D, dubinsLookup, width, height, 
          configurationSpace, visualization);
  // mark start as open
  start.open();
  start.sampling_table = &sampling_table;
  // This can not be deleted
  // push on priority queue aka open list
  O.push(&start);
  iPred = start.setIdx(width, height);
  nodes3D[iPred] = start;

  // NODE POINTER
  NodeAWS* nPred;
  NodeAWS* nSucc;

  // float max = 0.f;

  // continue until O empty
  while (!O.empty()) {
    // pop node with lowest cost from priority queue
    nPred = O.top();
    O_H.push(nPred);
    // pop node from the open list and start with a fresh node
    O.pop();
    // set index
    iPred = nPred->setIdx(width, height);
    iterations++;
    // RViz visualization
    if (Constants::visualization) {
      visualization.publishNode3DPoses(*nPred);
      visualization.publishNode3DPose(*nPred);
      // d.sleep();
    }
    // _____________________________
    // LAZY DELETION of rewired node
    // if there exists a pointer this node has already been expanded
    if (nodes3D[iPred].isClosed()) {
      continue;
    }

    // _________________
    // EXPANSION OF NODE
    if (nodes3D[iPred].isOpen()) {
      // add node to closed list
      nodes3D[iPred].close();
      // _________
      // GOAL TEST
      if (*nPred == goal || iterations > Constants::iterations) {
        // DEBUG
        if (iterations > Constants::iterations) std::cout << "iterations exceeded" << std::endl;
        else std::cout<<"goal found" <<std::endl;
        std::cout<<nPred->getPred()<< " , "<< nPred <<std::endl;
        
        // return nPred;
        return O_H.top();
      }

      // ____________________
      // CONTINUE WITH SEARCH
      // _______________________
      // SEARCH WITH DUBINS SHOT
      // TODO: a dubins algorithm that is suitable for AWS 
      // if (Constants::dubinsShot && nPred->isInRange(goal) && nPred->getPrim() < 0) {
      //   auto dubins_goal = dubinsShot(*nPred, goal, configurationSpace);
      //   std::cout << "dubinsShot: "<< dubins_goal << std::endl;
      //   if (dubins_goal != nullptr && *dubins_goal == goal) {
      //     return dubins_goal;
      //   }
      // }

      // ______________________________
      // SEARCH WITH FORWARD SIMULATION
      for (int i = 0; i < dir; ++i) {
        // create possible successor
        nPred->sampling_table = &sampling_table;
        nSucc = nPred->createSuccessor(i);
        if(nSucc==nPred){
          std::cout << "looping at start"<<std::endl;
          return nSucc;
        }
        // set index of the successor
        iSucc = nSucc->setIdx(width, height);
        // ensure successor is on grid and traversable
        if (!nSucc->isOnGrid(width, height) ||
            !configurationSpace.isTraversable(nSucc)) 
        {
          // std::cout<< "succ is not on grid and traversable" << std::endl;
          delete nSucc;
          continue;
        }
        // calculate new G value
        nSucc->updateG();
        newG = nSucc->getG(); // get updated g value
        // ensure successor is not on closed list or it has the same index as the predecessor
        // if successor not on open list or found a shorter way to the cell
        if (nodes3D[iSucc].isClosed() 
            && iPred != iSucc 
            && newG >= nodes3D[iSucc].getG())
        {
          // std::cout<< "succ closed or not same as Pred or with higher G" << std::endl;
          delete nSucc; 
          continue;
        }
        updateH(*nSucc, goal, nodes2D, dubinsLookup, width, height, configurationSpace, visualization);

        if(nSucc==nSucc->getPred()){
          std::cout << "looping after H update"<<std::endl;
          return nSucc;
        }
        if (iSucc == iPred && 
            nSucc->getC() > 
            nPred->getC() - Constants::tieBreaker)// + Constants::tieBreaker) 
        {
          // std::cout<< "succ in same cell with larger C" << std::endl;
          delete nSucc;
          continue;
        }
        if (nSucc->getPred() == nSucc) {
          std::cout << "looping in exploration"<<std::endl;
          return nullptr;
        }
        // put successor on open list
        nSucc->open();
        nodes3D[iSucc] = *nSucc;
        // O.push(&nodes3D[iSucc]);
        O.push(nSucc);
        // delete nSucc;
      }
    }
  }
  std::cout << "No candidate state, path not found" << std::endl;
  return nullptr;
}

//###################################################
//                                        2D A*
//###################################################
float aStar(Node2D& start,
            Node2D& goal,
            Node2D* nodes2D,
            int width,
            int height,
            CollisionDetection& configurationSpace,
            Visualize& visualization) {

  // PREDECESSOR AND SUCCESSOR INDEX
  int iPred, iSucc;
  float newG;

  // reset the open and closed list
  for (int i = 0; i < width * height; ++i) {
    nodes2D[i].reset();
  }

  // VISUALIZATION DELAY
  ros::Duration d(0.001);

  boost::heap::binomial_heap<Node2D*,
        boost::heap::compare<CompareNodes>> O;
  // update h value
  start.updateH(goal);
  // mark start as open
  start.open();
  // push on priority queue
  O.push(&start);
  iPred = start.setIdx(width);
  nodes2D[iPred] = start;

  // NODE POINTER
  Node2D* nPred;
  Node2D* nSucc;

  // continue until O empty
  while (!O.empty()) {
    // pop node with lowest cost from priority queue
    nPred = O.top();
    // set index
    iPred = nPred->setIdx(width);

    // _____________________________
    // LAZY DELETION of rewired node
    // if there exists a pointer this node has already been expanded
    if (nodes2D[iPred].isClosed()) {
      // pop node from the open list and start with a fresh node
      O.pop();
      continue;
    }
    // _________________
    // EXPANSION OF NODE
    else if (nodes2D[iPred].isOpen()) {
      // add node to closed list
      nodes2D[iPred].close();
      nodes2D[iPred].discover();

      // RViz visualization
      if (Constants::visualization2D) {
        visualization.publishNode2DPoses(*nPred);
        visualization.publishNode2DPose(*nPred);
        //        d.sleep();
      }

      // remove node from open list
      O.pop();

      // _________
      // GOAL TEST
      if (*nPred == goal) {
        return nPred->getG();
      }
      // ____________________
      // CONTINUE WITH SEARCH
      else {
        // _______________________________
        // CREATE POSSIBLE SUCCESSOR NODES
        for (int i = 0; i < Node2D::dir; i++) {
          // create possible successor
          nSucc = nPred->createSuccessor(i);
          // set index of the successor
          iSucc = nSucc->setIdx(width);

          // ensure successor is on grid ROW MAJOR
          // ensure successor is not blocked by obstacle
          // ensure successor is not on closed list
          if (nSucc->isOnGrid(width, height) &&  configurationSpace.isTraversable(nSucc) && !nodes2D[iSucc].isClosed()) {
            // calculate new G value
            nSucc->updateG();
            newG = nSucc->getG();

            // if successor not on open list or g value lower than before put it on open list
            if (!nodes2D[iSucc].isOpen() || newG < nodes2D[iSucc].getG()) {
              // calculate the H value
              nSucc->updateH(goal);
              // put successor on open list
              nSucc->open();
              nodes2D[iSucc] = *nSucc;
              O.push(&nodes2D[iSucc]);
              delete nSucc;
            } else { delete nSucc; }
          } else { delete nSucc; }
        }
      }
    }
  }

  // return large number to guide search away
  return 1000;
}

//###################################################
//                                         COST TO GO
//###################################################
void  updateH(Node3D& start, const Node3D& goal, Node2D* nodes2D, float* dubinsLookup, int width, int height, CollisionDetection& configurationSpace, Visualize& visualization) {
  float dubinsCost = 0;
  float reedsSheppCost = 0;
  float twoDCost = 0;
  float twoDoffset = 0;
  float eulaDis = 0;

  // if dubins heuristic is activated calculate the shortest path
  // constrained without obstacles
  if (Constants::dubins) {

    // ONLY FOR dubinsLookup
    //    int uX = std::abs((int)goal.getX() - (int)start.getX());
    //    int uY = std::abs((int)goal.getY() - (int)start.getY());
    //    // if the lookup table flag is set and the vehicle is in the lookup area
    //    if (Constants::dubinsLookup && uX < Constants::dubinsWidth - 1 && uY < Constants::dubinsWidth - 1) {
    //      int X = (int)goal.getX() - (int)start.getX();
    //      int Y = (int)goal.getY() - (int)start.getY();
    //      int h0;
    //      int h1;

    //      // mirror on x axis
    //      if (X >= 0 && Y <= 0) {
    //        h0 = (int)(helper::normalizeHeadingRad(M_PI_2 - t) / Constants::deltaHeadingRad);
    //        h1 = (int)(helper::normalizeHeadingRad(M_PI_2 - goal.getT()) / Constants::deltaHeadingRad);
    //      }
    //      // mirror on y axis
    //      else if (X <= 0 && Y >= 0) {
    //        h0 = (int)(helper::normalizeHeadingRad(M_PI_2 - t) / Constants::deltaHeadingRad);
    //        h1 = (int)(helper::normalizeHeadingRad(M_PI_2 - goal.getT()) / Constants::deltaHeadingRad);

    //      }
    //      // mirror on xy axis
    //      else if (X <= 0 && Y <= 0) {
    //        h0 = (int)(helper::normalizeHeadingRad(M_PI - t) / Constants::deltaHeadingRad);
    //        h1 = (int)(helper::normalizeHeadingRad(M_PI - goal.getT()) / Constants::deltaHeadingRad);

    //      } else {
    //        h0 = (int)(t / Constants::deltaHeadingRad);
    //        h1 = (int)(goal.getT() / Constants::deltaHeadingRad);
    //      }

    //      dubinsCost = dubinsLookup[uX * Constants::dubinsWidth * Constants::headings * Constants::headings
    //                                + uY *  Constants::headings * Constants::headings
    //                                + h0 * Constants::headings
    //                                + h1];
    //    } else {

    //if (Constants::dubinsShot && std::abs(start.getX() - goal.getX()) >= 10 && std::abs(start.getY() - goal.getY()) >= 10)*/
    //      // start
    //      double q0[] = { start.getX(), start.getY(), start.getT()};
    //      // goal
    //      double q1[] = { goal.getX(), goal.getY(), goal.getT()};
    //      DubinsPath dubinsPath;
    //      dubins_init(q0, q1, Constants::r, &dubinsPath);
    //      dubinsCost = dubins_path_length(&dubinsPath);

    ompl::base::DubinsStateSpace dubinsPath(Constants::r);
    State* dbStart = (State*)dubinsPath.allocState();
    State* dbEnd = (State*)dubinsPath.allocState();
    dbStart->setXY(start.getX(), start.getY());
    dbStart->setYaw(start.getT());
    dbEnd->setXY(goal.getX(), goal.getY());
    dbEnd->setYaw(goal.getT());
    dubinsCost = dubinsPath.distance(dbStart, dbEnd);
    if (Constants::time_measurement) dubinsCost = dubinsCost/Constants::max_wheel_v;
  }

  // if reversing is active use a
  if (Constants::reverse && !Constants::dubins) {
    //    ros::Time t0 = ros::Time::now();
    ompl::base::ReedsSheppStateSpace reedsSheppPath(Constants::r);
    State* rsStart = (State*)reedsSheppPath.allocState();
    State* rsEnd = (State*)reedsSheppPath.allocState();
    rsStart->setXY(start.getX(), start.getY());
    rsStart->setYaw(start.getT());
    rsEnd->setXY(goal.getX(), goal.getY());
    rsEnd->setYaw(goal.getT());
    reedsSheppCost = reedsSheppPath.distance(rsStart, rsEnd);
    if (Constants::time_measurement) reedsSheppCost = reedsSheppCost/Constants::max_wheel_v;
    //    ros::Time t1 = ros::Time::now();
    //    ros::Duration d(t1 - t0);
    //    std::cout << "calculated Reed-Sheep Heuristic in ms: " << d * 1000 << std::endl;
  }

  // if twoD heuristic is activated determine shortest path
  // unconstrained with obstacles
  if (Constants::twoD && !nodes2D[(int)start.getY() * width + (int)start.getX()].isDiscovered()) {
    //    ros::Time t0 = ros::Time::now();
    // create a 2d start node
    Node2D start2d(start.getX(), start.getY(), 0, 0, nullptr);
    // create a 2d goal node
    Node2D goal2d(goal.getX(), goal.getY(), 0, 0, nullptr);
    // run 2d astar and return the cost of the cheapest path for that node
    float astarCost = aStar(start2d, goal2d, nodes2D, width, height, configurationSpace, visualization);
    if (Constants::time_measurement) astarCost = astarCost/Constants::max_wheel_v;
    nodes2D[(int)start.getY() * width + (int)start.getX()].setG(astarCost);
    //    ros::Time t1 = ros::Time::now();
    //    ros::Duration d(t1 - t0);
    //    std::cout << "calculated 2D Heuristic in ms: " << d * 1000 << std::endl;
  }

  if (Constants::twoD) {
    // offset for same node in cell
    twoDoffset = sqrt(((start.getX() - (long)start.getX()) - (goal.getX() - (long)goal.getX())) * ((start.getX() - (long)start.getX()) - (goal.getX() - (long)goal.getX())) +
                      ((start.getY() - (long)start.getY()) - (goal.getY() - (long)goal.getY())) * ((start.getY() - (long)start.getY()) - (goal.getY() - (long)goal.getY())));
    if (Constants::time_measurement) twoDoffset = twoDoffset/Constants::max_wheel_v;
    twoDCost = nodes2D[(int)start.getY() * width + (int)start.getX()].getG() - twoDoffset;

  }

  if (Constants::eula_dis){
    eulaDis = sqrt((start.getX() - goal.getX()) * (start.getX() - goal.getX()) 
              + (start.getY() - goal.getY()) * (start.getY() - goal.getY()));
    if (Constants::time_measurement) eulaDis = eulaDis/Constants::max_wheel_v;
    twoDCost = eulaDis;
  }

  // return the maximum of the heuristics, making the heuristic admissable
  if (Constants::heading_heuristic) {
    float cost_yaw = std::abs(
      Helper::normalizeHeadingRad(start.getT() - goal.getT()));
    if(Constants::time_measurement) cost_yaw = 
              cost_yaw/Constants::max_steering_v;
    // start.setH(start.getH() + cost_yaw);
    twoDCost = 1.3*std::max((float)1.3*twoDCost , cost_yaw);
            //  + 0.1*std::min(twoDCost , cost_yaw);
    // twoDCost = 1.5*twoDCost + 0.1*cost_yaw;
  }
  // std::cout << "dubinsCost: "<< dubinsCost << " reedsSheppCost: "<< reedsSheppCost << " twoDCost: "<< twoDCost << std::endl;
  start.setH(std::max(reedsSheppCost, std::max(dubinsCost, twoDCost)));
}

//###################################################
//                                        DUBINS SHOT
//###################################################
Node3D* dubinsShot(Node3D& start, const Node3D& goal, CollisionDetection& configurationSpace) {
  // start
  double q0[] = { start.getX(), start.getY(), start.getT() };
  // goal
  double q1[] = { goal.getX(), goal.getY(), goal.getT() };
  // initialize the path
  DubinsPath path;
  // calculate the path
  dubins_init(q0, q1, Constants::r, &path);

  int i = 0;
  float x = 0.f;
  float length = dubins_path_length(&path);

  Node3D* dubinsNodes = new Node3D [(int)(length / Constants::dubinsStepSize) + 1];

  // avoid duplicate waypoint
  x += Constants::dubinsStepSize;
  while (x <  length) {
    double q[3];
    dubins_path_sample(&path, x, q);
    dubinsNodes[i].setX(q[0]);
    dubinsNodes[i].setY(q[1]);
    dubinsNodes[i].setT(Helper::normalizeHeadingRad(q[2]));

    // collision check
    if (configurationSpace.isTraversable(&dubinsNodes[i])) {

      // set the predecessor to the previous step
      if (i > 0) {
        dubinsNodes[i].setPred(&dubinsNodes[i - 1]);
      } else {
        dubinsNodes[i].setPred(&start);
      }

      if (&dubinsNodes[i] == dubinsNodes[i].getPred()) {
        std::cout << "looping shot";
      }

      x += Constants::dubinsStepSize;
      i++;
    } else {
      //      std::cout << "Dubins shot collided, discarding the path" << "\n";
      // delete all nodes
      delete [] dubinsNodes;
      return nullptr;
    }
  }

  //  std::cout << "Dubins shot connected, returning the path" << "\n";
  return &dubinsNodes[i - 1];
}


NodeAWS* dubinsShot(NodeAWS& start, const NodeAWS& goal, CollisionDetection& configurationSpace) {
  // start
  double q0[] = { start.getX(), start.getY(), start.getT() };
  // goal
  double q1[] = { goal.getX(), goal.getY(), goal.getT() };
  // initialize the path
  DubinsPath path;
  // calculate the path
  dubins_init(q0, q1, Constants::r, &path);

  int i = 0;
  float x = 0.f;
  float length = dubins_path_length(&path);

  NodeAWS* dubinsNodes = new NodeAWS [(int)(length / Constants::dubinsStepSize) + 1];

  // avoid duplicate waypoint
  x += Constants::dubinsStepSize;
  while (x <  length) {
    double q[3];
    dubins_path_sample(&path, x, q);
    dubinsNodes[i].setX(q[0]);
    dubinsNodes[i].setY(q[1]);
    dubinsNodes[i].setT(Helper::normalizeHeadingRad(q[2]));

    // collision check
    if (configurationSpace.isTraversable(&dubinsNodes[i])) {

      // set the predecessor to the previous step
      if (i > 0) {
        dubinsNodes[i].setPred(&dubinsNodes[i - 1]);
      } else {
        dubinsNodes[i].setPred(&start);
      }

      if (&dubinsNodes[i] == dubinsNodes[i].getPred()) {
        std::cout << "looping shot";
      }

      x += Constants::dubinsStepSize;
      i++;
    } else {
      //      std::cout << "Dubins shot collided, discarding the path" << "\n";
      // delete all nodes
      delete [] dubinsNodes;
      return nullptr;
    }
  }

  //  std::cout << "Dubins shot connected, returning the path" << "\n";
  return &dubinsNodes[i - 1];
}

template<typename T>
std::vector<float> linspace(T start_in, T end_in, int num_in)
{

  std::vector<float> linspaced;

  float start = static_cast<float>(start_in);
  float end = static_cast<float>(end_in);
  float num = static_cast<float>(num_in);

  if (num == 0) { return linspaced; }
  if (num == 1) 
    {
      linspaced.push_back(start);
      return linspaced;
    }

  float delta = (end - start) / (num - 1);

  for(int i=0; i < num-1; ++i)
    {
      linspaced.push_back(start + delta * i);
    }
  linspaced.push_back(end); // I want to ensure that start and end
                            // are exactly the same as the input
  return linspaced;
}

Eigen::Matrix<float,2,2> R(float theta){
  Eigen::Matrix<float,2,2> R;
  R << cos(theta), -sin(theta),
       sin(theta), cos(theta);
  return R;
}

int sign(double val) {
    if (val > 0) {
        return 1;
    } else if (val < 0) {
        return -1;
    } else { // val == 0
        return 0;
    }
}

std::vector<std::vector<float>> AWSConstructSamplingTable(int r_samples, int psi_samples, std::vector<float> omega_sample_list, float delta_cur){
  float epsilon = 1e-5;
  // float delta_theta = Constants::deltaHeadingRad;
  // float steering_limit = Constants::steering_limit;

  float delta_sample_r = M_PI_2/r_samples - epsilon;
  float delta_psi = 2*M_PI/psi_samples;

  std::vector<float> wheel_positions_x=Constants::wheel_positions_x;
  std::vector<float> wheel_positions_y=Constants::wheel_positions_y;

  std::vector<std::vector<float>> samplingTable;
  for (int i = 0; i <= r_samples; ++i){
    for (int j = 0; j <= psi_samples; ++j){
      float r_scale = tan(delta_sample_r*i + epsilon);
      float psi = -M_PI+delta_psi*j;
      float r_0 = -r_scale*cos(psi);
      float r_1 = -r_scale*sin(psi);
      bool infeasible = false;
      // check if the ICM position is feasible
      for (int k = 0; k < Constants::wheel_num; ++k){
        float steer_ulim_p_pi_d_2 = 
          Constants::wheel_steer_limits_up[k] + M_PI_2;
        float steer_llim_p_pi_d_2 = 
          Constants::wheel_steer_limits_low[k] + M_PI_2;
        float x = wheel_positions_x[k];
        float y = wheel_positions_y[k];
        float r_w_0 = r_0 + x;
        float r_w_1 = r_1 + y;

        if (Constants::has_steer_limit &&
            (r_w_0*sin(steer_ulim_p_pi_d_2) - r_w_1*cos(steer_ulim_p_pi_d_2))*
            (r_w_0*sin(steer_llim_p_pi_d_2) - r_w_1*cos(steer_llim_p_pi_d_2)) 
            > 0
            // abs(atan2(r_w_1,r_w_0))<steer_llim_p_pi_d_2 ||
            // abs(atan2(r_w_1,r_w_0))>steer_ulim_p_pi_d_2
            ){
          std::cout << "steer limit violated: "<< 
            atan2(r_w_1, r_w_0)/M_PI*180-90 <<r_scale<< std::endl;
          infeasible = true;
          break;
        }
      }
      if (infeasible)
      {
        continue;
      }
      
      for (float omega: omega_sample_list){
        float max_omega = delta_cur/r_scale;
        float prac_omega = sign(omega)*
                std::min(abs(omega), abs(max_omega));
        float v_x = -prac_omega*r_1;
        float v_y = prac_omega*r_0;
        Eigen::Vector2f r(r_0, r_1);
        Eigen::Vector2f d_xy = R(prac_omega)*r - r;
        float d_x = d_xy(0);
        float d_y = d_xy(1);
        std::vector<float> combo_pos = {d_x, d_y, prac_omega, 
                                        v_x, v_y, prac_omega};
        // std::vector<float> combo_neg = {d_x, d_y, omega,
        //                                 -v_x, -v_y, omega};
        infeasible = false;
        for (int m = 0; m < samplingTable.size(); ++m){
          if (abs(combo_pos[0]-samplingTable[m][0])<1e-1 &&
              abs(combo_pos[1]-samplingTable[m][1])<1e-1 &&
              abs(combo_pos[2]-samplingTable[m][2])<1e-1){
            infeasible = true;
            break;
          }
        }
        if (infeasible) continue;
        std::cout<<"combo_pos: "<<combo_pos[0]<<","
        <<combo_pos[0]<<","
        <<combo_pos[0]<<", steer limit not violated:" 
        << atan2(r_1, r_0)/M_PI*180-90<<", r:"<<r_scale<<std::endl;
        samplingTable.push_back(combo_pos);
        // samplingTable.push_back(combo_neg);
        if (omega == max_omega) break;
      }
    }
  }
  // std::unordered_set<std::vector<float>> uniqueTable(
  //                                 samplingTable.begin(),
  //                                 samplingTable.end()
  //                                 );
  std::vector<std::vector<float>> uniqueTable;
  std::unique_copy(samplingTable.begin(), 
                    samplingTable.end(), 
                    std::back_inserter(uniqueTable));
  samplingTable.assign(uniqueTable.begin(), uniqueTable.end());
  return samplingTable;
}