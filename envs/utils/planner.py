import math
import numpy as np
from collections import deque
from geometry_msgs.msg import Pose
from nav_msgs.msg import Path
from tf.transformations import euler_from_quaternion, quaternion_from_euler

class Waypoint:
    def __init__(self, pose: Pose):
        self.x = pose.position.x
        self.y = pose.position.y
        self.z = pose.position.z
        self.qx = pose.orientation.x
        self.qy = pose.orientation.y
        self.qz = pose.orientation.z
        self.qw = pose.orientation.w


class GlobalPlanner:
    def __init__(self, global_path_msg: Path):
        self.global_path_msg = global_path_msg

        self.original_global_waypoints = None
        # self.global_original_path = None
        self.global_path_waypoints = None
        self.interval_dis = 0  # total local target waypoints
        self.initialize_path()
        self.insert_checkout()
        # self.generate_global_path_waypoints()

    def initialize_path(self):
        """
        Generate Global Path in original distance.
        :return: None
        """
        self.original_global_waypoints = [Waypoint(wpt.pose) for wpt in self.global_path_msg.poses]
        print(f"Total length: {len(self.original_global_waypoints)}")


    def generate_global_path_waypoints(self):
        if self.original_global_waypoints is None:
            print(f"Empty Global Plan, Please Check !")
            return

        if len(self.original_global_waypoints) <= 2:
            print(f"Too short Global Plann !")
            return

        global_path_waypoints = []
        # step = math.ceil(len(self.original_global_waypoints) / self.interval_dis)
        # for i in range(self.interval_dis, len(self.original_global_waypoints), step):
        #     global_path_waypoints.append(self.original_global_waypoints[i])
        for i in range(len(self.original_global_waypoints)):
            global_path_waypoints.append(self.original_global_waypoints[i])
        self.global_path_waypoints = deque(global_path_waypoints)
        print(f"We totally choose {len(self.global_path_waypoints)} waypoints as our global path !")

        # for i in range(len(self.global_planner) - 1):
        #     dis = math.sqrt((self.global_planner[i+1].x - self.global_planner[i].x)**2 + (self.global_planner[i + 1].y - self.global_planner[i].y) ** 2)
        #     print(f"interval dis: {dis}")

    def insert_checkout(self):
        """
        I find an weird bug in carla-ros-bridge. That is in Town03, if the global route
        include a lane-change, the interval of some waypoints will be very Large, which would cause
        the STAgent unstable in route_bev. So, in this function, I'll insert some waypoints if the
        interval become too large.
        """
        if len(self.original_global_waypoints) == 0:
            print(f"Invalid Global Path !")
            return
        new_waypoints = []

        interval_max_threshold = 3
        target_interval = 1.0
        times = 0
        for idx in range(len(self.original_global_waypoints) - 1):
            current_wpt = self.original_global_waypoints[idx]
            next_wpt = self.original_global_waypoints[idx + 1]
            interval = np.linalg.norm(np.array([current_wpt.x, current_wpt.y, current_wpt.z]) - np.array(
                [next_wpt.x, next_wpt.y, next_wpt.z]))
            # print(f"interval : {interval}")
            if interval > interval_max_threshold:
                times += 1
                num = int(interval / target_interval)  # inserted wpts number .
                vx = next_wpt.x - current_wpt.x
                vy = next_wpt.y - current_wpt.y
                delta_x = target_interval * vx / interval
                delta_y = target_interval * vy / interval
                q = quaternion_from_euler(0, 0, np.arctan2(vy, vx))

                for id in range(num):
                    temp_pose = Pose()
                    temp_pose.position.x = current_wpt.x + delta_x * (id + 1)
                    temp_pose.position.y = current_wpt.y + delta_y * (id + 1)
                    temp_pose.position.z = current_wpt.z
                    temp_pose.orientation.x = q[0]
                    temp_pose.orientation.y = q[1]
                    temp_pose.orientation.z = q[2]
                    temp_pose.orientation.w = q[3]

                    temp_wpt = Waypoint(temp_pose)

                    new_waypoints.append(temp_wpt)

            else:
                new_waypoints.append(current_wpt)

        new_waypoints.append(self.original_global_waypoints[-1])

        print(f"Total invalid interval times: {times}")
        print(f"Original Global Waypoints: {len(self.original_global_waypoints)}")
        print(f"New Global Waypoints: {len(new_waypoints)}")

        self.original_global_waypoints = deque(new_waypoints)