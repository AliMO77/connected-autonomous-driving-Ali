import numpy as np
from scipy.spatial import KDTree

class WaypointsDatabase:
    """This class can be used to query the closest waypoint to a given (x,y) point"""
    def __init__(self, waypoints):
        self.waypoints = waypoints
        
        self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints]
        self.waypoint_tree = KDTree(self.waypoints_2d)


        
    def get_next_closest_idx(self, pose):


        # Find the closest waypoints to pose *that comes after pose on the track*
        # If pose is between x0 and x1, closer to x0, this should still return the index/distance of/to x1

        x, y = pose[0], pose[1]
        closest_idx = self.waypoint_tree.query([x, y], 1)[1]
        
        
        # Check if closest is ahead or behind the car
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx - 1]

        # Equation for hyperplane through closest_coords
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x, y])

        val = np.dot(cl_vect - prev_vect, pos_vect - cl_vect)


        # The closest_wp is behind the car, choose the next one
        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)

        

        return closest_idx
    

    def printer(self, pose):
        
        print(f"closest index: {pose} ")
        
        

