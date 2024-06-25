from enum import unique
from scipy.spatial import cKDTree
import numpy as np
from plan_utils.common import ROS_INTERFACE
from nav_msgs.msg import OccupancyGrid
import rospy


class MapUI:
    def __init__(self, ri: ROS_INTERFACE, map_topic="/map"):
        self.ri = ri
        self.index_map = None
        self.position_map = None
        self.kdtree = None
        
        self.ri.add_subscriber(map_topic, OccupancyGrid, self.map_callback, {})


    def get_closest_point(self, point):
        return self.map_data[self.tree.query(point)[1]]
    
    def map_callback(self, msg, dict_args):
        self.index_map = np.array(msg.data).reshape(
            msg.info.height, msg.info.width)
        self.map_height = msg.info.height
        self.map_width = msg.info.width
        occ = []
        for i in range(self.map_height):
            for j in range(self.map_width):
                if self.index_map[i][j] != 0:
                    occ.append([msg.info.origin.position.x + i * msg.info.resolution, msg.info.origin.position.y + j * msg.info.resolution])

        self.position_map = np.array(occ)
        self.kdtree = cKDTree(self.position_map)
        print('map updated')

    def get_occ_in_range(self, x, r):
        occ_index_lists = self.kdtree.query_ball_point(x, r)
        occ_point_lists = []
        for l in occ_index_lists:
            occ_point_lists.append(self.position_map[l])
        return occ_point_lists
    
    def get_occ_in_range_traj(self, x, r):
        occ_index_lists = self.kdtree.query_ball_point(x, r)
        unique_index = []
        for l in occ_index_lists:
            for i in l:
                if i not in unique_index:
                    unique_index.append(i)
        return self.position_map[unique_index]

        

if __name__ == "__main__":
    ri = ROS_INTERFACE()
    mu = MapUI(ri)
    while not rospy.is_shutdown():
        ri.rate.sleep()
        if mu.kdtree is not None:
            print(mu.get_occ_in_range(
                np.array([[1,1],[2,2],[3,3]]), 20))