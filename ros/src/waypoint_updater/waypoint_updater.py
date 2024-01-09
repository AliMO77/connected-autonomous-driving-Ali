#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
from WaypointsDatabase import WaypointsDatabase
import numpy as np
import math

'''
This node will publish waypoints ahead of the car's current position.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.
'''

class WaypointUpdater(object):

    ###########################################init##########################################
    def __init__(self):
        rospy.init_node('waypoint_updater',log_level=rospy.INFO)

        # Subscribe to input topics
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_callback)
        rospy.Subscriber('/base_waypoints', Lane, self.track_waypoints_callback)
        rospy.Subscriber("/traffic_waypoints", Int32, self.next_traffic_light_waypoint_callback)

        # Publisher for computed final waypoints
        self.final_waypoints_publisher = rospy.Publisher('final_waypoints', Lane, queue_size=1)
        self.N = 100 # Number of waypoints to publish (planning horizon)
        self.max_deceleration = 0.5
        self.base_waypoints = None

        # Wait until the required info has been received
        rate = rospy.Rate(50) # 50Hz loop
        self.current_car_position = None
        self.waypoints_db = None
        self.next_traffic_light_stopline_index = -1
        
        rospy.loginfo("waiting for initial waypoint")
        while not rospy.is_shutdown():
            if self.current_car_position is not None and self.waypoints_db is not None:
                break
            rate.sleep()
        # Main loop for the node, running at a fixed rate
        while not rospy.is_shutdown():
            self.process()
            rate.sleep()


    ###########################################   
    def pose_callback(self, msg: PoseStamped):
        self.current_car_position = np.array([msg.pose.position.x, msg.pose.position.y]) # XYZ position of the car
        

    def track_waypoints_callback(self, msg: Lane):
        self.waypoints_db = WaypointsDatabase(msg.waypoints)
    
    def next_traffic_light_waypoint_callback(self, msg: Int32):
        self.next_traffic_light_stopline_index = msg.data
    

    def make_lane(self):
        lane = Lane()
        
        cls_idx = self.waypoints_db.get_next_closest_idx(self.current_car_position)
        farthest_idx = cls_idx + self.N
        self.base_waypoints = self.waypoints_db.waypoints[cls_idx:farthest_idx]
        
        if self.next_traffic_light_stopline_index == -1 or (self.next_traffic_light_stopline_index >= farthest_idx):
            lane.waypoints = self.base_waypoints
        else:
            lane.waypoints = self.decelerate(self.base_waypoints,cls_idx)
            #rospy.logwarn("Traffic light waypoint: {}".format(self.next_traffic_light_stopline_index))
        #rospy.logwarn("Current position: {}, Closest waypoint: {}, farthest waypoint: {}".format(self.current_car_position, self.waypoints_db.waypoints_2d[cls_idx],self.waypoints_db.waypoints_2d[farthest_idx]))
        return lane
   
    def decelerate(self, waypoints, closest_idx):
        points = []

        for index, wp in enumerate(waypoints):
            point = Waypoint()
            point.pose = wp.pose
            #-3 to ensure the vehicle stops with its center before the stop line 
            stopping_idx = max(self.next_traffic_light_stopline_index - closest_idx - 3, 0)
            distance = self.get_distance(waypoints,index,stopping_idx)

            vel = math.sqrt(2 * self.max_deceleration * distance)
            # if vel is too small --> set to 0
            if vel < 1.0:
                vel = 0.0

            # Constraint to the speed limit
            point.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            points.append(point)
            #rospy.logwarn("next red light idx: {}, closest_idx: {} ,stopping_idx {}".format(self.next_traffic_light_stopline_index,closest_idx,stopping_idx,point.pose))

        return points
        
    def get_distance(self,waypoints,index,stopping_idx):
        
        dist = 0
        # calculate the Euclidean distance between these two points
        def calculate_euclidean_distance(a, b):
            return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)
        
        for i in range(index, stopping_idx):
            dist += calculate_euclidean_distance(waypoints[i].pose.pose.position, waypoints[i+1].pose.pose.position)
        
        return dist


    def get_wp_velocity(self,waypoint):
        return waypoint.twist.twist.linear.x

    def set_wp_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity    
    


    def process(self):
        # TODO: Use self.final_waypoints_pub to publish the next target waypoints
        # In phase 1: we can ignore traffic lights and simply output the next N waypoints *ahead of the car*, with their default velocity
        # In phase 2: you need to adjust target speeds on waypoints in order to smoothly brake until the car reaches the waypoint
        # corresponding to the next red light's stop line (stored in self.next_traffic_light_stopline_index, == -1 if no next traffic light).
        # Advice: make sure to complete dbw_node and have the car driving correctly while ignoring traffic lights before you tackle phase 2 
        
        lane = self.make_lane()
        
        self.final_waypoints_publisher.publish(lane)

    ###########################################main##########################################    

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except Exception as ex:
        rospy.logerr('Could not start waypoint updater node.')
        raise ex
