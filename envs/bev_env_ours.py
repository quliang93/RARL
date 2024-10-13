"""
Designed by quliang, 2023/10/6
Build an environment containing both static / dynamic BEV information.

data 10/6: bev-static & dynamic observation image;
Modified 1: add front empty check for invalid stop flag acquire .
Modified 2: check the propri-obs from CARLA-API to solve the failure of odometry in dynamic scenarios .
        about some details:
          For some weird reason, if I train the agent in Dynamic Scenarios, the odometry will fail to
          callback after some episodes, which is so weird and I totally have no idea about it. However,
          I find the carla and ros-bridge still working, so, in this version, I'll try to use CARLA-API
          to acquire the Location . Hope everything will be fine !

Modified 3: the most important change is the propri_obs real_time_orientation_diff, make it unsigned .
"""
import collections
import math
import random
import shapely
import cv_bridge
#import matplotlib.pyplot as plt
import cv2
import time
import carla
import rospy
import numpy as np
import message_filters
from tf import transformations
# from utils.bev_transformations import transform_objects_to_ego_vehicle_frame, yaw_to_rotation_matrix
from .utils.planner import GlobalPlanner, Waypoint
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Point32, PolygonStamped, Polygon, Point32, PointStamped, Pose, PoseStamped
from jsk_recognition_msgs.msg import PolygonArray, RectArray, Rect
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
from sensor_msgs.msg import LaserScan, Image, PointCloud, Imu
from derived_object_msgs.msg import Object, ObjectArray
from agents.tools.misc import is_within_distance, get_trafficlight_trigger_location
from carla_msgs.msg import CarlaEgoVehicleStatus, CarlaEgoVehicleControl, CarlaCollisionEvent, CarlaLaneInvasionEvent, CarlaEgoVehicleInfo
from .utils.bev_utils import BEVObject


# town02: 20 routes
scenarios_route = {
    "Town02_OLD_FASHION": [[70, -241.2, 0.22251926362514496,
                    0.0, 0.0, 0.0],
                   [-7.439950942993164, -129.5831298828125, 0.22239071130752563,
                    -0.0018313561089213074, -0.0016865819013735378, -0.6964543404141674, 0.7175967901924862]]
}


# town01: 20 routes
scenarios_route_backup1 = {
    "Town01_route1": [[269, -133.7, 0.22251926362514496,
                0.0, 0.0, 0.0],
               [301.7040100097656, -326.7915954589844, 0.002655200893059373,
                0.003347757514641976, -3.3680672550586575e-05, 0.9999453791224079, 0.009901017976930375]],
}

# town03: 20 routes
scenarios_route_town03 = {
    "Town03_route1": [[-118.1, -0.3, 0.22251926362514496,
                       0.0, 0.0, 0.0],
                      [-6.334556579589844, 49.08905792236328, 0.002910823794081807,
                       -0.002962325930724719, -0.00293927640743068, -0.7085522175340336, 0.7056460446334863]],
}


# town07: 20 routes
scenarios_route_backup7 = {
    "Town07_route1": [[-149.8, 43.5, 0.22251926362514496,
                       0.0, 0.0, -90.0],
                      [-1.2457358837127686, 87.64522552490234, 0.002748088678345084,
                       0.0025741037500423614, -0.0025890208301251505, 0.7048604369177823, 0.7093366164446174]],
}


class BEVENV:
    def __init__(self, bev_range, bev_img_size, ego_vehicle_dimensions, target_velocity, scenarios_random=True):
        rospy.init_node("bev_env")

        self.client = carla.Client("127.0.0.1", 2000)
        self.client.set_timeout(10.0)

        self.world = self.client.get_world()
        self.map = self.world.get_map()
        rospy.loginfo(f"World Got !")
        ego_vehicle_info = rospy.wait_for_message("/carla/ego_vehicle/vehicle_info", CarlaEgoVehicleInfo)
        # self.ego_id = ego_vehicle_info.id
        self.ego_vehicle_actor = self.world.get_actor(ego_vehicle_info.id)
        rospy.loginfo(f"Ego vehicle Actor Got !")

        # Route
        self.scenarios_route = scenarios_route["Town02_OLD_FASHION"]

        self.index = 0
        self.ego_location = None
        self.ego_heading = None
        self.ego_speed = None
        self.surrounding_objs = []
        self.ego_odom = None
        self.ego_vehicle_status = None
        self.cad_ranges = None
        self.bev_range = bev_range          # [x_low, y_low, x_high, y_high]
        self.bev_img_size = bev_img_size    # [H, W]
        self.next_wpt = None

        self.bev_obs_size = [128, 128]      # size of input bev img observation .

        x_range = self.bev_range[2] - self.bev_range[0]
        y_range = self.bev_range[3] - self.bev_range[1]

        self.ego_anchor_pixel = [int(self.bev_img_size[1] * self.bev_range[3] / y_range - 1),
                                 int(self.bev_img_size[0] * self.bev_range[2] / x_range - 1)]

        self.ego_vehicle_dimensions = ego_vehicle_dimensions    # [length, width]
        self.run_off_dis_threshold = 2.0          # m, max lateral distance threshold .
        self.angle_diff_max = 60  # in degree, more strictly limit about angle-diff between heading and waypoint .
        self.target_velocity = target_velocity
        self.min_velocity = 2  # m/s
        self.max_velocity = 10 # m/s
        self.voxel_size = [(bev_range[2]-bev_range[0])/self.bev_img_size[0],
                           (bev_range[3]-bev_range[1])/self.bev_img_size[1]]
        # to analyze whether the agent stop in invalid manner .
        # like no objects(ped, veh) nearby but stop for a long time .
        # for control-rate is around 10 Hz . so calculate it in 30 frames means invalid stop for 3 s.
        self.invalid_stop_frames = 0

        # Flag
        self.collision_victims = []
        self.collision_with_actor = None

        self.crossed_lane_markings = None

        self.bridge = CvBridge()

        # subscribers
        # odom_sub = message_filters.Subscriber("/carla/ego_vehicle/odometry", Odometry)
        odom_sub = message_filters.Subscriber("/carla/ego_vehicle/imu", Imu)
        vehicle_status_sub = message_filters.Subscriber("/carla/ego_vehicle/vehicle_status", CarlaEgoVehicleStatus)
        objs_sub = message_filters.Subscriber("/carla/ego_vehicle/objects", ObjectArray)
        cad_sub = message_filters.Subscriber("/cad_carla_gt", LaserScan)
        self.sys_synchronizer = message_filters.ApproximateTimeSynchronizer([odom_sub, vehicle_status_sub, objs_sub, cad_sub], queue_size=10, slop= 0.1)
        # self.sys_synchronizer.registerCallback(self.sys_callback)
        self.sys_synchronizer.registerCallback(self.sys_callback_api)
        collision_sub = rospy.Subscriber("/carla/ego_vehicle/collision", CarlaCollisionEvent,
                                         self.collision_callback)  # collision check
        lane_invasion_sub = rospy.Subscriber("/carla/ego_vehicle/lane_invasion", CarlaLaneInvasionEvent,
                                             self.lane_invasion_callback)  #

        # For visualization
        view_front_sub = rospy.Subscriber("/carla/ego_vehicle/rgb_view/image", Image, self.view_image_callback)

        # publishers
        # self.reset_pub = rospy.Publisher("/carla/ego_vehicle/control/set_transform", Pose, latch= True, queue_size= 5)
        self.global_goal_pub = rospy.Publisher("/carla/ego_vehicle/goal", PoseStamped, latch=True, queue_size=5)  # Set Goal WPT for Global Planning
        self.control_pub = rospy.Publisher("/carla/ego_vehicle/vehicle_control_cmd", CarlaEgoVehicleControl, latch= True, queue_size= 10) # for control
        self.render_pub = rospy.Publisher("/bev_perception", Image, queue_size= 10)  # for rendering .
        self.polygon_pub = rospy.Publisher("/cad_polygons", PolygonArray, queue_size= 3)
        self.local_path_pub = rospy.Publisher("/local_path", MarkerArray, queue_size= 1)


        self.run_step([0.0, 0.0])
        # self.place_ego_vehicle()
        self.place_ego_vehicle_api()
        self.send_goal()

        # Global Planning .
        global_path_msg = rospy.wait_for_message("/carla/ego_vehicle/waypoints", Path)
        rospy.loginfo("Global Waypoints GOT !")
        self.global_planner = GlobalPlanner(global_path_msg)
        self.global_planner.generate_global_path_waypoints()

        # when ever initialize the global route, popleft a wpt as the first wpt.
        self.next_wpt = self.global_planner.global_path_waypoints.popleft()

        """
        About Traffic Light
        """
        self._last_traffic_light = None

        self.color_mode = {
            4: (0, 0, 255), # Pedestrian
            6: (255, 0, 0), # Vehicle
            0: (0, 255, 0)  # ego vehicle
        }
        # while self.ego_odom is None or self.ego_vehicle_status is None:
        #     pass

        # Risk Aware
        self.temporal_risk_aware = None

        self.reset()


    def place_ego_vehicle(self):
        start_msg = Pose()
        start_msg.position.x = self.scenarios_route[0][0]       # 70.
        start_msg.position.y = self.scenarios_route[0][1]       # -241.
        start_msg.position.z = self.scenarios_route[0][2]       # 0.5
        start_msg.orientation.x = self.scenarios_route[0][3]   # 0.0
        start_msg.orientation.y = self.scenarios_route[0][4]   # -0.002859171052227825
        start_msg.orientation.z = self.scenarios_route[0][5]   # 0.0
        start_msg.orientation.w = self.scenarios_route[0][6]   # 0.9999959125620935
        self.reset_pub.publish(start_msg)
        rospy.sleep(0.15)
        # rospy.loginfo("*************** Start Pose ! ***************")


    def place_ego_vehicle_api(self):
        start_trans = carla.Transform()
        start_trans.location.x = self.scenarios_route[0][0]
        start_trans.location.y = -self.scenarios_route[0][1]
        start_trans.location.z = self.scenarios_route[0][2]

        start_trans.rotation.roll = self.scenarios_route[0][3]
        start_trans.rotation.pitch = self.scenarios_route[0][4]
        start_trans.rotation.yaw = self.scenarios_route[0][5]

        self.ego_vehicle_actor.set_transform(start_trans)


    def send_goal(self):
        """
        Work with carla-waypoints-publisher package to send the Global Planner A Goal.
        After doing this, you can subscribe /carla/ego_vehicle/waypoints to get global path.
        :return:
        """
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = "map"
        goal_msg.header.stamp = rospy.Time.now()
        goal_msg.pose.position.x = self.scenarios_route[1][0]    # 67.6239242553711
        goal_msg.pose.position.y = self.scenarios_route[1][1]    # -302.6101379394531
        goal_msg.pose.position.z = self.scenarios_route[1][2]    # 0.222742959856987
        goal_msg.pose.orientation.x = self.scenarios_route[1][3] # 0.003632720670779903
        goal_msg.pose.orientation.y = self.scenarios_route[1][4] # -8.561572205573734e-05
        goal_msg.pose.orientation.z = self.scenarios_route[1][5] # 0.9997136807333468
        goal_msg.pose.orientation.w = self.scenarios_route[1][6] # 0.023650635616409444
        self.global_goal_pub.publish(goal_msg)
        rospy.sleep(0.15)
        rospy.loginfo("*************** Goal Set ! ***************")


    def transform_map_to_local(self, map_points, ego_self_pos: np.ndarray, ego_self_yaw):
        """
        This function transform all points from map frame to Ego-Vehicle's frame.
        :param map_points: np.ndarray, (N, 2), points in map frame.
        :param ego_self_pos: np.ndarray, (1, 2), ego vehicle's center position in map frame.
        :param ego_self_yaw: the ego vehicle's current yaw angle
        :return: np.ndarray, (N, 2), shape the same as map_points
        """
        transform_matrix = np.array([[ np.cos(ego_self_yaw), np.sin(ego_self_yaw)],
                                     [-np.sin(ego_self_yaw), np.cos(ego_self_yaw)]], dtype= np.float32)
        map_points = map_points.T # (2, N)
        map_points[0, :] -= ego_self_pos[0]
        map_points[1, :] -= ego_self_pos[1]
        local_points = transform_matrix @ map_points
        return local_points.T


    def transform_ego_to_wpt(self, nearest_wpt: np.ndarray, ego_vehicle_location):
        """
        Transform the ego location point into the nearest waypoint's Frame .
        :param nearest_wpt: np.ndarray, (x, y, z, qx, qy, qz, qw)
        :param ego_vehicle_location: (x, y, z)
        :return: float, float: Y-axis distance, means the lateral distance to Route ., X-axis distance
        """
        _, _, wpt_yaw = transformations.euler_from_quaternion(nearest_wpt[3:])
        wpt_location = nearest_wpt[:2].reshape(2, 1)
        transform_matrix = np.array([[ np.cos(wpt_yaw), np.sin(wpt_yaw)],
                                     [-np.sin(wpt_yaw), np.cos(wpt_yaw)]], dtype= np.float32)
        ego_vehicle_point = ego_vehicle_location[:2].reshape(2, 1)
        ego_in_wpt = transform_matrix @ (ego_vehicle_point - wpt_location)
        ego_in_wpt = ego_in_wpt.T

        return np.abs(ego_in_wpt[0][1]), ego_in_wpt[0][0]


    def transform_coords_to_pixels(self, local_coords):
        """
        This function transform the local points into image space, pixel format.
        :param local_coords: np.ndarray: (N, 2)
        :return: local points' pixels in image, np.ndarray, (N, 2)
        """

        temp = local_coords.copy()
        temp[:, 0] = -local_coords[:, 1] / self.voxel_size[0]
        temp[:, 1] = -local_coords[:, 0] / self.voxel_size[1]

        temp[:, 0] += self.ego_anchor_pixel[0]
        temp[:, 1] += self.ego_anchor_pixel[1]
        temp = temp.astype(np.int32)
        return temp


    def draw_ego_bbox_img(self, input_img):
        """
        This function
        :param input_img:
        :return:
        """
        # ept_img = np.zeros((self.bev_img_size[0], self.bev_img_size[1], 3), dtype=np.uint8)
        delta_y = self.ego_vehicle_dimensions[0] / self.voxel_size[0] / 2
        delta_x = self.ego_vehicle_dimensions[1] / self.voxel_size[1] / 2 # delta x -> length

        cv2.rectangle(input_img, (self.ego_anchor_pixel[0] + int(delta_x), self.ego_anchor_pixel[1] + int(delta_y)),
                               (self.ego_anchor_pixel[0] - int(delta_x), self.ego_anchor_pixel[1] - int(delta_y)), self.color_mode[0], -1)

        return input_img


    def sys_callback(self, odom_msg: Odometry,
                     vehicle_status_msg: CarlaEgoVehicleStatus,
                     objs_msg: ObjectArray,
                     cad_msg: LaserScan):
        """
        sys_callback is the main callback for time synchronizer
        :param odom_msg: odometry
        :param vehicle_status_msg: vehicle status msg, including steering, throttle
        :param objs_msg: all surrounding objects (Veh / Ped) in MAP frame
        :param cad_msg:  cad perception results in ego vehicle frame
        :return: None
        """
        ego_pos = np.array([odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y])

        if len(objs_msg.objects) == 0:
            self.surrounding_objs = []

        surrounding_objs = []
        # pick out the surrounding objects less in 30m.
        for obj in objs_msg.objects:
            obj_pos = np.array([obj.pose.position.x, obj.pose.position.y])
            distance = np.linalg.norm(ego_pos - obj_pos)
            if distance < 30:
                surrounding_objs.append(obj)

        self.surrounding_objs = surrounding_objs      # list of Object .
        self.ego_odom = odom_msg                      # Odometry .
        self.ego_vehicle_status = vehicle_status_msg  # ego vehicle status .
        self.cad_ranges = cad_msg.ranges

        self.ego_location = np.array(
            [self.ego_odom.pose.pose.position.x, self.ego_odom.pose.pose.position.y, self.ego_odom.pose.pose.position.z])  # map frame

        self.ego_heading = np.array([self.ego_odom.pose.pose.orientation.x, self.ego_odom.pose.pose.orientation.y,
                                     self.ego_odom.pose.pose.orientation.z, self.ego_odom.pose.pose.orientation.w])  # map frame

        self.ego_speed = [self.ego_odom.twist.twist.linear.x, self.ego_odom.twist.twist.angular.z]
        # check if we arrive the next waypoint
        if self.next_wpt is None:
            return
        # ego_location, _, _, _ = self.get_propriceptive_obs() # if we use function here, we update all self.ego_xxx variable.

        next_wpt_np = np.array([self.next_wpt.x, self.next_wpt.y, self.next_wpt.z,
                                self.next_wpt.qx, self.next_wpt.qy, self.next_wpt.qz, self.next_wpt.qw], dtype= np.float32)
        _, dis = self.transform_ego_to_wpt(next_wpt_np, self.ego_location)
        # dis >= - threshold, the bigger the threshold is, the further the next wpt is .
        while dis >= -1.0 and len(self.global_planner.global_path_waypoints) >= 1:
            self.next_wpt = self.global_planner.global_path_waypoints.popleft()
            next_wpt_np = np.array([self.next_wpt.x, self.next_wpt.y, self.next_wpt.z,
                                    self.next_wpt.qx, self.next_wpt.qy, self.next_wpt.qz, self.next_wpt.qw],
                                   dtype=np.float32)
            _, dis = self.transform_ego_to_wpt(next_wpt_np, self.ego_location)

        # dis_to_next_wpt = np.linalg.norm(np.array([self.next_wpt.x, self.next_wpt.y]) - ego_location[:2])
        # if dis_to_next_wpt < 1 and len(self.global_planner.global_path_waypoints) >= 1: # threshold for arriving the waypoint .
        #     self.next_wpt = self.global_planner.global_path_waypoints.popleft()



    def sys_callback_api(self, imu_msg: Imu,
                     vehicle_status_msg: CarlaEgoVehicleStatus,
                     objs_msg: ObjectArray,
                     cad_msg: LaserScan):
        # print(f"start receiving the message ...")
        ego_loc = self.ego_vehicle_actor.get_location()
        ego_pos = np.array([ego_loc.x, -ego_loc.y])

        if len(objs_msg.objects) == 0:
            self.surrounding_objs = []

        surrounding_objs = []
        # pick out the surrounding objects less in 30m.
        for obj in objs_msg.objects:
            obj_pos = np.array([obj.pose.position.x, obj.pose.position.y])
            distance = np.linalg.norm(ego_pos - obj_pos)
            if distance < 30:
                surrounding_objs.append(obj)

        self.surrounding_objs = surrounding_objs      # list of Object .
        self.ego_odom = Odometry()
        self.ego_odom.pose.pose.position.x = ego_loc.x
        self.ego_odom.pose.pose.position.y = -ego_loc.y
        self.ego_odom.pose.pose.position.z = ego_loc.z
        self.ego_odom.pose.pose.orientation.x = vehicle_status_msg.orientation.x
        self.ego_odom.pose.pose.orientation.y = vehicle_status_msg.orientation.y
        self.ego_odom.pose.pose.orientation.z = vehicle_status_msg.orientation.z
        self.ego_odom.pose.pose.orientation.w = vehicle_status_msg.orientation.w
        self.ego_odom.twist.twist.linear.x = vehicle_status_msg.velocity
        self.ego_odom.twist.twist.angular.z = imu_msg.angular_velocity.z

        self.ego_vehicle_status = vehicle_status_msg  # ego vehicle status .
        self.cad_ranges = cad_msg.ranges

        self.ego_location = np.array(
            [self.ego_odom.pose.pose.position.x, self.ego_odom.pose.pose.position.y,
             self.ego_odom.pose.pose.position.z])  # map frame

        # print(f"current ego location: {self.ego_location}")

        self.ego_heading = np.array([self.ego_odom.pose.pose.orientation.x, self.ego_odom.pose.pose.orientation.y,
                                     self.ego_odom.pose.pose.orientation.z,
                                     self.ego_odom.pose.pose.orientation.w])  # map frame

        self.ego_speed = [self.ego_odom.twist.twist.linear.x, self.ego_odom.twist.twist.angular.z]

        # check if we arrive the next waypoint
        if self.next_wpt is None:
            return
        # ego_location, _, _, _ = self.get_propriceptive_obs() # if we use function here, we update all self.ego_xxx variable.

        next_wpt_np = np.array([self.next_wpt.x, self.next_wpt.y, self.next_wpt.z,
                                self.next_wpt.qx, self.next_wpt.qy, self.next_wpt.qz, self.next_wpt.qw],
                               dtype=np.float32)
        _, dis = self.transform_ego_to_wpt(next_wpt_np, self.ego_location)
        # dis >= - threshold, the bigger the threshold is, the further the next wpt is .
        while dis >= -1.0 and len(self.global_planner.global_path_waypoints) >= 1:
            self.next_wpt = self.global_planner.global_path_waypoints.popleft()
            next_wpt_np = np.array([self.next_wpt.x, self.next_wpt.y, self.next_wpt.z,
                                    self.next_wpt.qx, self.next_wpt.qy, self.next_wpt.qz, self.next_wpt.qw],
                                   dtype=np.float32)
            _, dis = self.transform_ego_to_wpt(next_wpt_np, self.ego_location)


    def collision_callback(self, collision_msg: CarlaCollisionEvent):
        """
        Callback for collision event check .
        :param collision_msg:
        """
        self.collision_with_actor = collision_msg.other_actor_id


    def lane_invasion_callback(self, lane_invasion_msg: CarlaLaneInvasionEvent):
        """
        Callback for lane invasion check
        :param lane_invasion_msg:
        """
        self.crossed_lane_markings = lane_invasion_msg.crossed_lane_markings


    def get_propriceptive_obs(self):
        """
        The propriceptive observation of the Ego-Vehicle,
        specially related to the speed, heading, etc.
        :return:
            ego_location: np.ndarray, (1, 3): [x, y, z]
            ego_heading:  np.ndarray, (1, 4): quaterion, [x, y, z, w]
            ego_speed:    np.ndarray, (1, 2): velocity, [linear, angular]
            proprioceptive obs: np.ndarray, [1, 6]: [throttle, steer, linear_vel, angular_vel, lateral_dis, orientation_angle_diff]
        """

        real_time_throttle = self.ego_vehicle_status.control.throttle              # real-time throttle, [-1, 1]
        real_time_steer = self.ego_vehicle_status.control.steer                    # real-time steer, [-1, 1]
        real_time_linear_vel = self.ego_odom.twist.twist.linear.x                  # real-time linear_vel,
        real_time_angular_vel = self.ego_odom.twist.twist.angular.z                # real-time angular_vel

        # calculate the lateral distance:
        wpts_array = np.array([[wpt.x, wpt.y, wpt.z, wpt.qx, wpt.qy, wpt.qz, wpt.qw]
                               for wpt in self.global_planner.original_global_waypoints], dtype=np.float32)
        distances = np.linalg.norm(wpts_array[:, :3] - self.ego_location, axis=1)
        nearest_wpt = wpts_array[np.argmin(distances)]
        real_time_lateral_dis, _ = self.transform_ego_to_wpt(nearest_wpt, self.ego_location)

        # calculate the orientation cosine similarity:
        _, _, ego_yaw = transformations.euler_from_quaternion(self.ego_heading)
        _, _, wpt_yaw = transformations.euler_from_quaternion([self.next_wpt.qx, self.next_wpt.qy, self.next_wpt.qz, self.next_wpt.qw])

        # if you want to test the diff-angle in degree, do (rad * 180 / np.pi)
        real_time_orientation_diff = (wpt_yaw - ego_yaw) * 180 / np.pi # angle-diff between wpt's and agent's heading. (rad unit)
        real_time_orientation_diff = real_time_orientation_diff if np.abs(real_time_orientation_diff) < 180 \
            else np.sign(real_time_orientation_diff)*(360.0 - np.abs(real_time_orientation_diff))
        real_time_orientation_diff = np.abs(real_time_orientation_diff) # make the orientation-diff unsigned .
        proprioceptive_obs = np.array([real_time_throttle, real_time_steer, real_time_linear_vel,
                                       real_time_angular_vel, real_time_lateral_dis, real_time_orientation_diff])

        # print(f"current ego location: {self.ego_location}")
        # print(f"current ego heading: {self.ego_heading}")
        # print(f"current ego speed: {self.ego_speed}")
        # print(f"current propriceptive_obs: {proprioceptive_obs}")

        return self.ego_location, self.ego_heading, self.ego_speed, proprioceptive_obs


    def get_cad_bev(self):
        """
        To convert CAD perception into BEV img， TOO Low .
        In this function, we don't use the two transform functions,
        Because if we do transform first and then convert the coords into pixels,
        all things seem work incorrectly, weird.
        :return: CAD BEV image, (H, W, 1) single-channel image
        """
        if self.cad_ranges is None:
            rospy.loginfo("No CAD perception !")
            return
        bev_img = np.zeros((self.bev_img_size[0], self.bev_img_size[1], 3), dtype= np.uint8)

        cad_points = []

        for idx in range(len(self.cad_ranges)):
            r = self.cad_ranges[idx] / self.voxel_size[0]
            theta = (0.5 + idx) * (2*np.pi/384) - np.pi
            cad_points.append([-r * np.sin(theta), -r * np.cos(theta)])

        cad_points = np.array(cad_points, dtype=np.int32)

        cad_points[:, 0] = cad_points[:, 0] + self.ego_anchor_pixel[0]
        cad_points[:, 1] = cad_points[:, 1] + self.ego_anchor_pixel[1]

        cv2.drawContours(bev_img, [cad_points], -1, (150, 150, 150), -1)
        # cad_bev_img_msg = self.bridge.cv2_to_imgmsg(bev_img, encoding= "bgr8")
        # self.cad_render_pub.publish(cad_bev_img_msg)

        return bev_img


    def get_detection_bev(self):
        """
        To generalize the BEV image of detection-based methods.
        :return: detection bev img, (H, W, 3)
        """
        if self.surrounding_objs is None:
            rospy.loginfo("Invalid Perception !")
            return

        bev_img = np.zeros((self.bev_img_size[0], self.bev_img_size[1], 3), dtype=np.uint8) # (H, W, n_channels)
        bev_img = self.draw_ego_bbox_img(bev_img) # draw self-ego vehicle bbox .
        location, quaternion, _, _ = self.get_propriceptive_obs()
        _, _, ego_self_yaw = transformations.euler_from_quaternion(quaternion)
        for object in self.surrounding_objs:
            obj_cls_id = object.classification
            obj_pose = object.pose
            obj_twist = object.twist
            # obj_accel = object.accel
            length, width, height = object.shape.dimensions
            bev_obj = BEVObject(obj_pose, obj_twist, length, width, obj_cls_id)
            bev_vertexes_map = bev_obj.get_four_vertexes_in_map()
            local_vertexes_ego = self.transform_map_to_local(bev_vertexes_map, location[:2], ego_self_yaw)
            vertexes_pixels = self.transform_coords_to_pixels(local_vertexes_ego)
            vertexes_pixels = vertexes_pixels.astype(np.int32)
            painting_color = self.color_mode[4] if obj_cls_id == 4 else self.color_mode[6]
            cv2.drawContours(bev_img, [vertexes_pixels], -1, painting_color, -1)

        # statics_objs_msg = self.bridge.cv2_to_imgmsg(bev_img, encoding= "bgr8")
        # self.static_objs_render_pub.publish(statics_objs_msg)
        # print("publishing")
        return bev_img


    def get_route_bev(self):
        """
        Get the route BEV map. Note the width of route in map should equal to vehicle.
        Same as the get_cad_bev(), we don't use the two transform functions here .
        :return: route map . (H, W, 3), display the ego self bbox .
        """
        # 1. transform the global waypoints into ego vehicle frame.
        location, quaternion, _, _ = self.get_propriceptive_obs()
        _ , _, yaw = transformations.euler_from_quaternion(quaternion)
        route_waypoints_map = []
        for wpt in list(self.global_planner.global_path_waypoints):
            local_x = (wpt.x - location[0]) * np.cos(yaw) + (wpt.y - location[1]) * np.sin(yaw)
            local_y = -(wpt.x - location[0]) * np.sin(yaw) + (wpt.y - location[1]) * np.cos(yaw)
            img_coord_x = -local_y / self.voxel_size[1] + self.ego_anchor_pixel[0]
            img_coord_y = -local_x / self.voxel_size[1] + self.ego_anchor_pixel[1]
            route_waypoints_map.append([img_coord_x, img_coord_y])

        route_local = np.array(route_waypoints_map, dtype= np.int32)
        # 2. paint Route on the image.
        route_bev_img = np.zeros((self.bev_img_size[0], self.bev_img_size[1], 3), dtype=np.uint8)
        cv2.polylines(route_bev_img, [route_local], False, (128, 0, 128), 40) # Not closed
        route_bev_img = self.draw_ego_bbox_img(route_bev_img)

        # 3. Publish image
        # route_bev_msg = self.bridge.cv2_to_imgmsg(route_bev_img, encoding="bgr8")
        # self.route_render_pub.publish(route_bev_msg)
        # print(f"Route BEV image publishing ...")
        return route_bev_img


    def polygons_overlap(self, poly1, poly2):
        poly1 = shapely.Polygon(poly1)
        poly2 = shapely.Polygon(poly2)
        return poly1.intersects(poly2)


    def get_dynamic_bev(self):
        """
        The most important Observation for RL input.
        In this bev-img, we visualize the dynamic information about objects around the ego vehicle.
        How to do it ?
        Maybe we should calculate the predicted traj. in map.
        And transform the predicted traj. from map to ego-vehicle frame.
        After we subscribe the topic "/carla/ego_vehicle/objects",
        we acquire all dynamic info. of surrounding objects.
        Then we predict the traj. of the object, then we transform the predicted traj.
        from map frame into ego_vehicle frame.
        Finally, we visualize the predicted traj. into the BEV image.
        :return:
        """
        self.temporal_risk_aware = False
        if self.surrounding_objs is None:
            rospy.loginfo("Invalid Perception !")
            return

        location, quaternion, _, _ = self.get_propriceptive_obs()
        _, _, ego_vehicle_yaw = transformations.euler_from_quaternion(quaternion)

        dynamic_bev_img = np.zeros((self.bev_img_size[0], self.bev_img_size[1], 3), dtype=np.uint8)

        # publish image message ...
        ego_obj_pose = self.ego_odom.pose.pose
        ego_obj_twist = self.ego_odom.twist.twist
        ego_obj_length = self.ego_vehicle_dimensions[0]
        ego_obj_width = self.ego_vehicle_dimensions[1]
        ego_obj_cls_id = 0
        bev_ego_obj = BEVObject(ego_obj_pose, ego_obj_twist, ego_obj_length, ego_obj_width, ego_obj_cls_id)
        ego_polygon_predicted = bev_ego_obj.predict_obj_polygon_vertexes()
        _, _, ego_vehicle_self_yaw = transformations.euler_from_quaternion(self.ego_heading)
        ego_local_coords = self.transform_map_to_local(ego_polygon_predicted, location[:2], ego_vehicle_yaw)
        ego_polygon_pixels = self.transform_coords_to_pixels(ego_local_coords)
        cv2.fillPoly(dynamic_bev_img, [ego_polygon_pixels], (0, 255, 0))

        for object in self.surrounding_objs:
            # temp_object = Object()
            obj_cls_id = object.classification
            obj_pose = object.pose
            obj_twist = object.twist
            obj_accel = object.accel
            length, width, height = object.shape.dimensions
            bev_obj = BEVObject(obj_pose, obj_twist, length, width, obj_cls_id)
            bev_pred_traj = bev_obj.predict_obj_polygon_vertexes() # but in map frame
            # we should transform the vertexes into ego-vehicle frame.
            obj_vertexes_local = self.transform_map_to_local(bev_pred_traj, location[:2], ego_vehicle_yaw)
            # transform vertexes (ego vehicle frame) into image pixels
            polygon_pixels = self.transform_coords_to_pixels(obj_vertexes_local)

            if self.polygons_overlap(ego_local_coords, obj_vertexes_local):
                self.temporal_risk_aware = True
                # rospy.loginfo(f"Potential collision detected with object {obj_cls_id} !!!")


            if bev_obj.cls_id == 4:
                cv2.fillPoly(dynamic_bev_img, [np.array(polygon_pixels, dtype= np.int32)], (0, 0, 255))
            else:
                cv2.fillPoly(dynamic_bev_img, [np.array(polygon_pixels, dtype=np.int32)], (255, 0, 0))



        # dynamic_bev_msg = self.bridge.cv2_to_imgmsg(dynamic_bev_img, encoding="bgr8")
        # self.dynamic_objs_render_pub.publish(dynamic_bev_msg)
        # print(f"Publishing predicted bev image message ...")

        return dynamic_bev_img


    def get_bev_img_obs(self):
        """
        This function get the bev obs for RL training
        :return: np.ndarray (10, 128, 128)
        """
        route_bev_img = self.get_route_bev()
        detections_bev_img = self.get_detection_bev()
        dynamic_bev_img = self.get_dynamic_bev()
        cad_bev_img = self.get_cad_bev()
        cad_bev_img_s = cv2.cvtColor(cad_bev_img, cv2.COLOR_RGB2GRAY)

        assert route_bev_img.shape[:2] == (512, 512)
        assert detections_bev_img.shape[:2] == (512, 512)
        assert dynamic_bev_img.shape[:2] == (512, 512)
        assert cad_bev_img_s.shape[:2] == (512, 512)
        stacked_img = np.zeros((512, 512, 10), dtype= np.float32)
        stacked_img[:, :, :3] = route_bev_img
        stacked_img[:, :, 3:6] = detections_bev_img
        stacked_img[:, :, 6:9] = dynamic_bev_img
        stacked_img[:, :, 9] = cad_bev_img_s

        # bev_resized = cv2.resize(stacked_img, (224, 224))
        bev_resized = cv2.resize(stacked_img, (self.bev_obs_size[0], self.bev_obs_size[1]))
        bev_resized = bev_resized / 255.0
        bev = np.transpose(bev_resized, (2, 0, 1))
        return bev


    def get_collision_state(self):
        """
        Get collision state .
        :return:
        """
        # rospy.loginfo(f"The Ego Vehicle collide with the actor: {self.collision_with_actor}")
        # if we collide with a new actor, we add it into a Record List of Victims.
        # if self.collision_with_actor not in self.collision_victims:
        #     self.collision_victims.append(self.collision_with_actor)
        if self.collision_with_actor == None:
            return False
        else:
            return True


    def get_lane_state(self):
        """
        Get lane invasion state .
        :return:
        """
        if self.crossed_lane_markings != None:
            if self.crossed_lane_markings[0] == 10:
                self.crossed_lane_markings = None
                return 2 # means cross the sidewalk
            else:
                self.crossed_lane_markings = None
                return 1

        else:
            return 1


    def get_run_off_lateral_distance(self):
        """
        Get run-off lateral distance to Global Route .
        :return: float
        """
        # 1. find the current nearest waypoint to ego vehicle
        wpts_array = np.array([[wpt.x, wpt.y, wpt.z, wpt.qx, wpt.qy, wpt.qz, wpt.qw]
                               for wpt in self.global_planner.original_global_waypoints], dtype= np.float32)
        ego_vehicle_location, _, _, _ = self.get_propriceptive_obs()
        distances = np.linalg.norm(wpts_array[:, :3]-ego_vehicle_location, axis= 1)
        nearest_wpt = wpts_array[np.argmin(distances)]
        # 2. transform the ego vehicle location into the Nerest Waypoint Frame.
        run_dis, _ = self.transform_ego_to_wpt(nearest_wpt, ego_vehicle_location)
        # 3 . return the y-axis distance .
        return run_dis


    def cosine_similarity(self, x1, y1, x2, y2):
        # 将点a和点b表示为NumPy数组
        vector_a = np.array([x1, y1])
        vector_b = np.array([x2, y2])

        # 计算点积
        dot_product = np.dot(vector_a, vector_b)

        # 计算向量的模
        norm_a = np.linalg.norm(vector_a)
        norm_b = np.linalg.norm(vector_b)
        assert norm_b != 0
        assert norm_a != 0
        # 计算余弦相似度
        cosine_sim = dot_product / (norm_a * norm_b)
        return cosine_sim, norm_b


    def get_consine_similarity(self):
        """
        Get the cosine similarity between the ego pose to next waypoint .
        :return: cosine, distance
        """
        ego_location, ego_heading, _, _ = self.get_propriceptive_obs()
        _, _, ego_yaw = transformations.euler_from_quaternion(ego_heading)
        next_wpt_np = np.array([self.next_wpt.x, self.next_wpt.y]).reshape(1, 2)
        local_wpt = self.transform_map_to_local(next_wpt_np, ego_location, ego_yaw)
        local_wpt = local_wpt[0]
        cosine, distance = self.cosine_similarity(1., 0., local_wpt[0], local_wpt[1])
        return cosine, distance


    def target_lane_empty_check(self):
        """
        This function will check the target lane whether it is empty.
        True: means at this moment, there are no obstacles stuck ahead of the target Route.
        False: means at this moment, there are some obstacles STUCK the Route in front of the agent !
        :return: True / False
        """
        # if there is no objects nearby, the target lane should be empty
        if len(self.surrounding_objs) == 0:
            return True

        # if the target lane wpts left less than N wpts, regard it as empty
        if len(self.global_planner.global_path_waypoints) <= 10:
            return True

        next_target_lane_area = list(self.global_planner.global_path_waypoints) #[:10]
        next_target_lane_area = next_target_lane_area[:10] # added 2024/1/23

        target_lane_wpts = np.array([(wpt.x, wpt.y) for wpt in next_target_lane_area]).reshape(-1, 2)

        obstacles = np.array([(obj.pose.position.x, obj.pose.position.y) for obj in self.surrounding_objs]).reshape(-1, 2)

        distance_matrix = np.sqrt(np.sum((obstacles.reshape(-1, 1, 2) - target_lane_wpts)**2, axis= -1))

        # min_distance = distance_matrix.min()
        min_index = np.unravel_index(distance_matrix.argmin(), distance_matrix.shape)
        # print(f"distance matrix shape: {distance_matrix.shape}")
        # print(f"min index: {min_index}, Current surrounding obstacles: {len(self.surrounding_objs)}")
        assert min_index[0] <= len(self.surrounding_objs)

        if len(self.surrounding_objs) <= min_index[0]:
            return True

        min_obstacle = self.surrounding_objs[min_index[0]]
        min_waypoint = next_target_lane_area[min_index[1]]

        min_waypoint = np.array([min_waypoint.x, min_waypoint.y, min_waypoint.z, min_waypoint.qx, min_waypoint.qy, min_waypoint.qz, min_waypoint.qw])
        min_obstacle = np.array([min_obstacle.pose.position.x, min_obstacle.pose.position.y, min_obstacle.pose.position.z])
        lat, lon = self.transform_ego_to_wpt(min_waypoint, min_obstacle)


        if min_index[1] == 0:
            # if exist behind the nearest next wpt, return True, that means key obstacles in behind .
            if lon <= 0:
                return True # emtpy road
            if lat <= 1.5:
                return False # busy road
            else:
                return True # empty road

        else:
            # just judge the lateral distance, if less than the road width, return False, else return True .
            if lat <= 1.5:
                return False
            else:
                return True


    def render(self):
        """
        Render four bev images.
        :return: None .
        """
        bev_cad = self.get_cad_bev()
        bev_route = self.get_route_bev()
        bev_detections = self.get_detection_bev()
        bev_dynamic_detections = self.get_dynamic_bev()

        render_img = np.zeros((self.bev_img_size[0], self.bev_img_size[1], 3), dtype= np.uint8)
        render_img[:256, :256, :] = cv2.resize(bev_route, (256, 256))
        render_img[:256, 256:, :] = cv2.resize(bev_detections, (256, 256))
        render_img[256:, :256, :] = cv2.resize(bev_cad, (256, 256))
        render_img[256:, 256:, :] = cv2.resize(bev_dynamic_detections, (256, 256))

        cv2.line(render_img, (0, 256), (512, 256), (128, 128, 128), 10)
        cv2.line(render_img, (256, 0), (256, 512), (128, 128, 128), 10)
        render_img_msg = self.bridge.cv2_to_imgmsg(render_img, encoding="bgr8")
        self.render_pub.publish(render_img_msg)
        # print("Rendering ...")


    def render_observation(self, step, store=True):
        """
        Render four bev images.
        :return: None .
        """
        bev_cad = self.get_cad_bev()
        bev_route = self.get_route_bev()
        bev_detections = self.get_detection_bev()
        bev_dynamic_detections = self.get_dynamic_bev()

        render_img = np.zeros((self.bev_img_size[0], self.bev_img_size[1], 3), dtype= np.uint8)
        render_img[:256, :256, :] = cv2.resize(bev_route, (256, 256))
        render_img[:256, 256:, :] = cv2.resize(bev_detections, (256, 256))
        render_img[256:, :256, :] = cv2.resize(bev_cad, (256, 256))
        render_img[256:, 256:, :] = cv2.resize(bev_dynamic_detections, (256, 256))

        cv2.line(render_img, (0, 256), (512, 256), (128, 128, 128), 10)
        cv2.line(render_img, (256, 0), (256, 512), (128, 128, 128), 10)
        render_img_msg = self.bridge.cv2_to_imgmsg(render_img, encoding="bgr8")
        if store:
            cv2.imwrite(f"visualize/obs/{str(step).zfill(6)}.png", render_img)
            # cv2.imwrite(f"visualize/scenarios/{str(step).zfill(6)}.png", self.view_img)
            rospy.loginfo(f"storing image ...")

        self.render_pub.publish(render_img_msg)
        # print("Rendering ...")


    def render_show(self, step, store=False):
        bev_cad = self.get_cad_bev()
        bev_route = self.get_route_bev()
        bev_dynamic_detections = self.get_dynamic_bev()

        render_img = cv2.resize(bev_dynamic_detections, (256, 256)) # render image
        # mask 1 about where to draw dynamic
        bev_dynamic_detections_s = cv2.cvtColor(cv2.resize(bev_dynamic_detections, (256, 256)), cv2.COLOR_RGB2GRAY)
        dynamic_mask = bev_dynamic_detections_s
        dynamic_mask = dynamic_mask.astype(np.uint8)
        zero_indices = lambda arr: arr == 0

        # mask 2 about where to draw cad area
        cad_bev_img_s = cv2.cvtColor(cv2.resize(bev_cad, (256, 256)), cv2.COLOR_RGB2GRAY)
        cad_mask = cad_bev_img_s / 255
        # print(cad_mask.max())
        # cad_mask = cad_mask.astype(np.uint8)
        nonzero_indices = lambda arr: arr != 0
        render_img[nonzero_indices(cad_mask) & zero_indices(dynamic_mask)] = (150, 150, 150)


        # mask 3 about where to draw route
        bev_route_s = cv2.cvtColor(cv2.resize(bev_route, (256, 256)), cv2.COLOR_RGB2GRAY)
        bev_route_s = bev_route_s.astype(np.uint8)
        # print(bev_route_s.max())
        render_img[nonzero_indices(bev_route_s) & zero_indices(dynamic_mask)] = (128, 0, 128)


        if store and self.view_img is not None:
            cv2.imwrite(f"visualize/bevobs/{str(step).zfill(6)}.png", render_img)
            cv2.imwrite(f"visualize/scenarios/{str(step).zfill(6)}.png", self.view_img)
            rospy.loginfo(f"storing image ...")


        render_img_msg = self.bridge.cv2_to_imgmsg(render_img, encoding="bgr8")
        self.render_pub.publish(render_img_msg)


    def render_show_visualize(self, step, store=False):
        bev_cad = self.get_cad_bev()
        bev_route = self.get_route_bev()
        bev_dynamic_detections = self.get_dynamic_bev()

        render_img = cv2.resize(bev_dynamic_detections, (256, 256)) # render image
        # mask 1 about where to draw dynamic
        bev_dynamic_detections_s = cv2.cvtColor(cv2.resize(bev_dynamic_detections, (256, 256)), cv2.COLOR_RGB2GRAY)
        dynamic_mask = bev_dynamic_detections_s
        dynamic_mask = dynamic_mask.astype(np.uint8)
        zero_indices = lambda arr: arr == 0

        # mask 2 about where to draw cad area
        cad_bev_img_s = cv2.cvtColor(cv2.resize(bev_cad, (256, 256)), cv2.COLOR_RGB2GRAY)
        cad_mask = cad_bev_img_s / 255
        # print(cad_mask.max())
        cad_mask = cad_mask.astype(np.uint8)
        nonzero_indices = lambda arr: arr != 0
        render_img[nonzero_indices(cad_mask) & zero_indices(dynamic_mask)] = (150, 150, 150)

        # mask 3 about where to draw route
        # bev_route_s = cv2.cvtColor(cv2.resize(bev_route, (256, 256)), cv2.COLOR_RGB2GRAY)
        # bev_route_s = bev_route_s.astype(np.uint8)
        # # print(bev_route_s.max())
        # render_img[nonzero_indices(bev_route_s) & zero_indices(dynamic_mask)] = (128, 0, 128)


        if store and self.view_img is not None:
            cv2.imwrite(f"visualize/bevobs/{str(step).zfill(6)}.png", render_img)
            cv2.imwrite(f"visualize/scenarios/{str(step).zfill(6)}.png", self.view_img)
            rospy.loginfo(f"storing image ...")

        # # draw cad polygon
        # cad_polygon_array = PolygonArray()
        # cad_polygon_array.header.stamp = rospy.Time.now()
        # cad_polygon_array.header.frame_id = "ego_vehicle"
        #
        # polygon_stamped = PolygonStamped()
        # polygon_stamped.header.stamp = rospy.Time.now()
        # polygon_stamped.header.frame_id = "ego_vehicle"
        #
        # for idx in range(len(self.cad_ranges)):
        #     r = self.cad_ranges[idx]
        #     theta = (0.5 + idx) * (2 * np.pi / 384) - np.pi
        #     point_local = Point32()
        #     point_local.x = r * np.cos(theta)
        #     point_local.y = r * np.sin(theta)
        #     point_local.z = 0.2
        #     polygon_stamped.polygon.points.append(point_local)
        #
        # cad_polygon_array.polygons.append(polygon_stamped)
        #
        # self.polygon_pub.publish(cad_polygon_array)
        #
        # # draw local path
        # marker_array_msg = MarkerArray()
        # location, quaternion, _, _ = self.get_propriceptive_obs()
        # _, _, yaw = transformations.euler_from_quaternion(quaternion)
        #
        # if len(self.global_planner.global_path_waypoints) != 0:
        #
        #     for index in range(len(self.global_planner.global_path_waypoints)):
        #         wpt = self.global_planner.global_path_waypoints[index]
        #         local_x = (wpt.x - location[0]) * np.cos(yaw) + (wpt.y - location[1]) * np.sin(yaw)
        #         local_y = -(wpt.x - location[0]) * np.sin(yaw) + (wpt.y - location[1]) * np.cos(yaw)
        #         mark = Marker()
        #         mark.header.stamp = rospy.Time.now()
        #         mark.header.frame_id = "ego_vehicle"
        #         mark.id = index
        #         mark.type = 2
        #         mark.pose.position.x = local_x
        #         mark.pose.position.y = local_y
        #         mark.pose.position.z = 0.4
        #         mark.pose.orientation.w = 1
        #         mark.scale.x = 0.3
        #         mark.scale.y = 0.3
        #         mark.scale.z = 0.3
        #         mark.color.b = 128
        #         mark.color.r = 128
        #         mark.color.a = 0.8
        #
        #         marker_array_msg.markers.append(mark)
        #
        #     self.local_path_pub.publish(marker_array_msg)



        render_img_msg = self.bridge.cv2_to_imgmsg(render_img, encoding="bgr8")
        self.render_pub.publish(render_img_msg)



    def render_spatial_temporal(self, step, store=False):
        bev_cad = self.get_cad_bev()
        # bev_route = self.get_route_bev()
        bev_dynamic_detections = self.get_dynamic_bev()

        render_img_temporal = cv2.resize(bev_dynamic_detections, (256, 256)) # render image

        render_img_spatial =  np.zeros((256, 256, 3), dtype= np.uint8) # cv2.resize(bev_cad, (256, 256))  # render image
        cad_bev_img_s = cv2.cvtColor(cv2.resize(bev_cad, (256, 256)), cv2.COLOR_RGB2GRAY)
        cad_mask = cad_bev_img_s / 255

        cad_mask = cad_mask.astype(np.uint8)
        nonzero_indices = lambda arr: arr != 0
        render_img_spatial[nonzero_indices(cad_mask)] = (150, 150, 150)

        if store and self.view_img is not None:
            cv2.imwrite(f"visualize/spatial/{str(step).zfill(6)}.png", render_img_spatial)
            cv2.imwrite(f"visualize/temporal/{str(step).zfill(6)}.png", render_img_temporal)
            rospy.loginfo(f"storing image ...")

        # render_img_msg = self.bridge.cv2_to_imgmsg(render_img, encoding="bgr8")
        # self.render_pub.publish(render_img_msg)


    def render_rviz(self, step, store=False):
        # bev_cad = self.get_cad_bev()
        # bev_route = self.get_route_bev()
        # bev_dynamic_detections = self.get_dynamic_bev()
        #
        # render_img = cv2.resize(bev_dynamic_detections, (256, 256)) # render image
        # # mask 1 about where to draw dynamic
        # bev_dynamic_detections_s = cv2.cvtColor(cv2.resize(bev_dynamic_detections, (256, 256)), cv2.COLOR_RGB2GRAY)
        # dynamic_mask = bev_dynamic_detections_s
        # dynamic_mask = dynamic_mask.astype(np.uint8)
        # zero_indices = lambda arr: arr == 0
        #
        # # mask 2 about where to draw cad area
        # cad_bev_img_s = cv2.cvtColor(cv2.resize(bev_cad, (256, 256)), cv2.COLOR_RGB2GRAY)
        # cad_mask = cad_bev_img_s / 255
        # # print(cad_mask.max())
        # cad_mask = cad_mask.astype(np.uint8)
        # nonzero_indices = lambda arr: arr != 0
        # render_img[nonzero_indices(cad_mask) & zero_indices(dynamic_mask)] = (150, 150, 150)
        #
        # # mask 3 about where to draw route
        # # bev_route_s = cv2.cvtColor(cv2.resize(bev_route, (256, 256)), cv2.COLOR_RGB2GRAY)
        # # bev_route_s = bev_route_s.astype(np.uint8)
        # # # print(bev_route_s.max())
        # # render_img[nonzero_indices(bev_route_s) & zero_indices(dynamic_mask)] = (128, 0, 128)
        #
        #
        # if store and self.view_img is not None:
        #     cv2.imwrite(f"visualize/bevobs/{str(step).zfill(6)}.png", render_img)
        #     cv2.imwrite(f"visualize/scenarios/{str(step).zfill(6)}.png", self.view_img)
        #     rospy.loginfo(f"storing image ...")

        # draw cad polygon
        time_start = time.time()
        # cad_polygon_array = PolygonArray()
        # cad_polygon_array.header.stamp = rospy.Time.now()
        # cad_polygon_array.header.frame_id = "ego_vehicle"
        #
        # polygon_stamped = PolygonStamped()
        # polygon_stamped.header.stamp = rospy.Time.now()
        # polygon_stamped.header.frame_id = "ego_vehicle"
        #
        # for idx in range(len(self.cad_ranges)):
        #     r = self.cad_ranges[idx]
        #     theta = (0.5 + idx) * (2 * np.pi / 384) - np.pi
        #     point_local = Point32()
        #     point_local.x = r * np.cos(theta)
        #     point_local.y = r * np.sin(theta)
        #     point_local.z = 0.2
        #     polygon_stamped.polygon.points.append(point_local)
        #
        # cad_polygon_array.polygons.append(polygon_stamped)
        #
        # self.polygon_pub.publish(cad_polygon_array)
        time2 = time.time()

        # draw local path
        marker_array_msg = MarkerArray()
        location, quaternion, _, _ = self.get_propriceptive_obs()
        _, _, yaw = transformations.euler_from_quaternion(quaternion)

        if len(self.global_planner.global_path_waypoints) > 2:
            for index in range(len(self.global_planner.global_path_waypoints)):
                wpt = self.global_planner.global_path_waypoints[index]
                local_x = (wpt.x - location[0]) * np.cos(yaw) + (wpt.y - location[1]) * np.sin(yaw)
                local_y = -(wpt.x - location[0]) * np.sin(yaw) + (wpt.y - location[1]) * np.cos(yaw)
                mark = Marker()
                mark.header.stamp = rospy.Time.now()
                mark.header.frame_id = "ego_vehicle"
                mark.id = index
                mark.type = 2
                mark.pose.position.x = local_x
                mark.pose.position.y = local_y
                mark.pose.position.z = 0.4
                mark.pose.orientation.w = 1
                mark.scale.x = 0.3
                mark.scale.y = 0.3
                mark.scale.z = 0.3
                mark.color.b = 128
                mark.color.r = 128
                mark.color.a = 0.8

                marker_array_msg.markers.append(mark)

            self.local_path_pub.publish(marker_array_msg)

        time3 = time.time()

        rospy.loginfo(f"cad cost time: {time2 - time_start}")
        rospy.loginfo(f"local path cost time: {time3 - time2}")

        # render_img_msg = self.bridge.cv2_to_imgmsg(render_img, encoding="bgr8")
        # self.render_pub.publish(render_img_msg)


    def render_velocity(self):
        """
        This function is designed for visualizing the current velocity from ego odometry.

        """

        _, _, _, propri_state = self.get_propriceptive_obs()
        linear_velocity = propri_state[2] * 3.6 # km/h
        # fig, ax = plt.subplots()
        self.vel_ax.cla()

        self.vel_ax.set_xlim(0, 162)
        self.vel_ax.set_ylim(0, 18)
        self.vel_ax.set_xlabel('step (frame)')
        self.vel_ax.set_ylabel('speed (km/h)')
        self.vel_ax.set_xticks([])
        self.vel_ax.grid(True, linestyle='--')

        # color
        self.vel_fig.patch.set_facecolor('black')
        self.vel_ax.set_facecolor('black')
        self.vel_ax.spines['bottom'].set_color('white')
        self.vel_ax.spines['top'].set_color('white')
        self.vel_ax.spines['right'].set_color('white')
        self.vel_ax.spines['left'].set_color('white')
        self.vel_ax.xaxis.label.set_color('white')
        self.vel_ax.yaxis.label.set_color('white')
        self.vel_ax.tick_params(axis='x', colors='white')
        self.vel_ax.tick_params(axis='y', colors='white')


        self.history_velocity.append(linear_velocity)

        self.vel_ax.plot(self.history_velocity, color='#DCCEC2', linewidth=2.5)
        self.vel_fig.canvas.draw()
        img = np.frombuffer(self.vel_fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(self.vel_fig.canvas.get_width_height()[::-1] + (3,))
        vel_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        render_img_msg = self.bridge.cv2_to_imgmsg(vel_img, encoding="bgr8")
        self.render_pub.publish(render_img_msg)


    def view_image_callback(self, img_msg):
        try:
            self.view_img = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")

        except cv_bridge.CvBridgeError as e:
            rospy.logerr(e)



    def render_obs_action(self, action):
        """
            This aims for rendering both bev observation & action
        """
        # Create a white image
        steer = action[0] * np.abs(action[0])  # steering
        accel = action[1] if action[1] > 0.2 else -1

        # self.view_img shape : (800, 600)
        if self.view_img is None:
            return

        # 1. drawing steer panel.
        steer_start = (300, 520)
        cv2.ellipse(self.view_img, center=steer_start, axes=(70, 70), angle=180, startAngle=0, endAngle=180, color=(222, 122, 84),
                    thickness=4, lineType=cv2.LINE_AA)
        angular_x = steer_start[0] - int(64 * np.sin(-steer * 1.5707))
        angular_y = steer_start[1] - int(64 * np.cos(-steer * 1.5707))
        steer_end = (angular_x, angular_y)
        cv2.line(self.view_img, steer_start, steer_end, (0, 255, 255), 4)

        # 2. drawing throttle & brake panel.
        accel_start = (500, 500)
        bar_length = int(100 * np.abs(accel))

        cv2.rectangle(self.view_img, (accel_start[0]-3, accel_start[1]-3), (accel_start[0] + 103, accel_start[1]+14), (222, 122, 84), thickness=4)
        cv2.rectangle(self.view_img, (accel_start[0]-3, accel_start[1]-3), (accel_start[0] - 103, accel_start[1]+14), (222, 122, 84), thickness=4)

        cv2.putText(self.view_img, "-1", (accel_start[0] - 100 -10, accel_start[1]+30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(self.view_img, "0", (accel_start[0] - 7, accel_start[1]+30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(self.view_img, "+1", (accel_start[0] + 100 - 10, accel_start[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 2)

        if accel > 0:
            cv2.rectangle(self.view_img, accel_start, (accel_start[0] + bar_length, accel_start[1]+11), (0, 255, 0, 40), -1)
        else:
            cv2.rectangle(self.view_img, accel_start, (accel_start[0] - bar_length, accel_start[1]+11), (0, 0, 255, 40), -1)


        steer_text = f"steer={steer:.2f}"
        accel_text = f"accel={accel:.2f}"
        cv2.putText(self.view_img, steer_text, (steer_start[0]-50, 560), cv2.FONT_HERSHEY_COMPLEX,
                    0.7, (0, 0, 0), 2)
        cv2.putText(self.view_img, accel_text, (accel_start[0]-50, 560), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                    (0, 0, 0), 2)


        # render observation
        bev_cad = self.get_cad_bev()
        bev_route = self.get_route_bev()
        bev_dynamic_detections = self.get_dynamic_bev()

        render_img = cv2.resize(bev_dynamic_detections, (256, 256)) # render image
        # mask 1 about where to draw dynamic
        bev_dynamic_detections_s = cv2.cvtColor(cv2.resize(bev_dynamic_detections, (256, 256)), cv2.COLOR_RGB2GRAY)
        dynamic_mask = bev_dynamic_detections_s
        dynamic_mask = dynamic_mask.astype(np.uint8)
        zero_indices = lambda arr: arr == 0

        # mask 2 about where to draw cad area
        cad_bev_img_s = cv2.cvtColor(cv2.resize(bev_cad, (256, 256)), cv2.COLOR_RGB2GRAY)
        cad_mask = cad_bev_img_s / 255
        # print(cad_mask.max())
        cad_mask = cad_mask.astype(np.uint8)
        nonzero_indices = lambda arr: arr != 0
        render_img[nonzero_indices(cad_mask) & zero_indices(dynamic_mask)] = (150, 150, 150)

        # mask 3 about where to draw route
        bev_route_s = cv2.cvtColor(cv2.resize(bev_route, (256, 256)), cv2.COLOR_RGB2GRAY)
        bev_route_s = bev_route_s.astype(np.uint8)
        # print(bev_route_s.max())
        render_img[nonzero_indices(bev_route_s) & zero_indices(dynamic_mask)] = (128, 0, 128)


        self.view_img[0: 256, 800-256: 800] = cv2.resize(render_img, (256, 256))

        cv2.rectangle(self.view_img, (800-256, 0), (800, 256), (222, 122, 84), thickness=4)


        img_msg = self.bridge.cv2_to_imgmsg(self.view_img, encoding='bgr8')

        self.render_pub.publish(img_msg)


    def run_step_backup(self, action):
        """
        Control the ego vehicle in carla-way (steering, accel)
        :param action: (accel, steering), both belongs to [-1, 1]
        :return: None
        """
        carla_control_msg= CarlaEgoVehicleControl()
        # carla_control_msg.steer = action[0] # steering
        # Here we times steer * np.abs(steer) to make turning more softly
        carla_control_msg.steer = action[0] * np.abs(action[0])  # steering
        if action[1] > 0:
            carla_control_msg.throttle = action[1]
            carla_control_msg.brake = 0.0
        else:
            carla_control_msg.throttle = 0.0
            carla_control_msg.brake = action[1]

        self.control_pub.publish(carla_control_msg)
        rospy.sleep(0.15)


    def run_step(self, action):
        """
        Control the ego vehicle in carla-way (steering, accel), global clip accel
        :param action: (accel, steering), both belongs to [-1, 1]
        :return: None
        """
        carla_control_msg= CarlaEgoVehicleControl()
        # carla_control_msg.steer = action[0] # steering
        # Here we times steer * np.abs(steer) to make turning more softly
        carla_control_msg.steer = action[0] * np.abs(action[0])  # steering
        # carla_control_msg.steer = action[0] # * np.abs(action[0])  # steering
        if action[1] > 0.2: # 0.2 for normal speed, 0.1 for adversarial events
            carla_control_msg.throttle = action[1]
            carla_control_msg.brake = 0.0
        else:
            carla_control_msg.throttle = 0.0
            carla_control_msg.brake = np.abs(action[1]) # action[1] #

        self.control_pub.publish(carla_control_msg)
        rospy.sleep(0.15)



    def reward_mark_XIV(self, step):
        """
            Budapest's Thesis Reward Style .
            :param step:
            :return:
        """
        done = False
        result = 0
        reward_sidewalk = 0
        reward_collision = 0
        reward_mixed = 0
        reward_invalid_stop = 0
        reward_smooth = 0
        reward_deviation = 0
        _, _, _, propri_obs = self.get_propriceptive_obs()  # acquire propri-obs
        lane = self.get_lane_state()
        collision = self.get_collision_state()

        current_throttle = propri_obs[0]
        current_steer = propri_obs[1]

        # smooth
        # if np.abs(current_steer - self.previous_steer) > 0.3:
        #     reward_smooth = -0.05 # if steer change to sharply, we penalize such actions.

        if np.abs(current_steer - self.previous_steer) > 0.3:
            # reward_smooth = -0.3 * np.abs(current_steer - self.previous_steer) # if steer change to sharply, we penalize such actions.
            reward_smooth = -0.07
        curr_vel = propri_obs[2]  # current vel
        curr_dis_lateral = propri_obs[4]  # current lateral distance to lane
        curr_diff_angle = propri_obs[5]  # current diff angle between wpt and agent

        if curr_dis_lateral >= 1.05:
            reward_deviation = -0.1

        alpha_rew = 1 - np.abs(curr_diff_angle / self.angle_diff_max) if np.abs(
            curr_diff_angle) < self.angle_diff_max else 0
        d_norm = 1 / 3  # 1 / self.run_off_dis_threshold

        # check if there are obstacles on the next 10 target waypoints (around 10 m)
        empty_flag = self.target_lane_empty_check()

        if curr_vel < 0.3 and empty_flag == True:  # and len(self.surrounding_objs) == 0
            self.invalid_stop_frames += 1
        else:
            self.invalid_stop_frames = 0

        if self.invalid_stop_frames > 100:
            done = True
            reward_invalid_stop = -10
            result = "invalid-stop"

        # different reward acquire in different velocity mode.
        if curr_vel < self.min_velocity:
            reward_mixed = curr_vel / self.min_velocity * (1 - d_norm) * alpha_rew

        if curr_vel >= self.target_velocity - 3 and curr_vel < self.target_velocity + 3:
            reward_mixed = 1 * (1 - d_norm) * alpha_rew

        if curr_vel >= self.target_velocity + 3:
            reward_mixed = (1 - ((curr_vel - self.target_velocity) / (self.max_velocity - self.target_velocity))) * (
                    1 - d_norm) * alpha_rew

        if curr_vel >= self.max_velocity:  # over-speed
            reward_mixed = -10  # (self.max_velocity - curr_vel) * 3.6
            done = True
            result = "over-speed"

        if collision:
            done = True
            reward_collision = -10
            result = "collision"

        if lane == 2 and step < 8: print(f"reset is on Sidewalk line: {lane}")
        if lane == 1:
            # result = "Cross center line"
            pass
        elif lane == 2:
            done = True
            reward_sidewalk = -10
            result = "Cross Sidewalk line"

        if curr_dis_lateral > self.run_off_dis_threshold:
            done = True
            result = "run-off"

        R = reward_collision + reward_sidewalk + reward_mixed + reward_invalid_stop + reward_smooth + reward_deviation

        # ego_vehicle_location = self.ego_vehicle_actor.get_location()

        if len(self.global_planner.global_path_waypoints) == 0:
            done = True
            result = "Goal"

        if step >= 700:
            done = True
            result = "Time out"

        # print(f"****************************** STEP : {step} *************************************")
        # print(f"Delta steer compared to last steer : {np.abs(current_steer - self.previous_steer)}")
        # print(f"Delta angle diff between ego and wpt: {curr_diff_angle}")
        # print(f"Delta lateral distance between ego and wpt: {curr_dis_lateral}")
        # print(f"Reward collision: {reward_collision}")
        # print(f"Reward sidewalk: {reward_sidewalk}")
        # print(f"Reward mixed: {reward_mixed}")
        # print(f"Reward invalid stop: {reward_invalid_stop}")
        # print(f"Reward smooth: {reward_smooth}")
        # print(f"Reward deviation: {reward_deviation}")
        # print(f"**********************************************************************************")


        return R, done, result



    def reward_mark_XIV_reverse(self, step):
        """
            Budapest's Thesis Reward Style .
            :param step:
            :return:
        """
        done = False
        result = 0
        reward_sidewalk = 0
        reward_collision = 0
        reward_mixed = 0
        reward_invalid_stop = 0
        reward_smooth = 0
        reward_deviation = 0
        _, _, _, propri_obs = self.get_propriceptive_obs()  # acquire propri-obs
        lane = self.get_lane_state()
        collision = self.get_collision_state()

        current_throttle = propri_obs[0]
        current_steer = propri_obs[1]

        # smooth
        # if np.abs(current_steer - self.previous_steer) > 0.3:
        #     reward_smooth = -0.05 # if steer change to sharply, we penalize such actions.

        if np.abs(current_steer - self.previous_steer) > 0.3:
            # reward_smooth = -0.3 * np.abs(current_steer - self.previous_steer) # if steer change to sharply, we penalize such actions.
            reward_smooth = -0.07
        curr_vel = propri_obs[2]  # current vel
        curr_dis_lateral = propri_obs[4]  # current lateral distance to lane
        curr_diff_angle = propri_obs[5]  # current diff angle between wpt and agent

        if curr_dis_lateral >= 1.05:
            reward_deviation = -0.1

        alpha_rew = 1 - np.abs(curr_diff_angle / self.angle_diff_max) if np.abs(
            curr_diff_angle) < self.angle_diff_max else 0
        d_norm = 1 / 3  # 1 / self.run_off_dis_threshold

        # check if there are obstacles on the next 10 target waypoints (around 10 m)
        empty_flag = self.target_lane_empty_check()

        if curr_vel < 0.3 and empty_flag == True:  # and len(self.surrounding_objs) == 0
            self.invalid_stop_frames += 1
        else:
            self.invalid_stop_frames = 0

        if self.invalid_stop_frames > 100:
            done = True
            reward_invalid_stop = -10
            result = "invalid-stop"

        # different reward acquire in different velocity mode.
        if curr_vel < self.min_velocity:
            reward_mixed = curr_vel / self.min_velocity * (1 - d_norm) * alpha_rew

        if curr_vel >= self.target_velocity - 3 and curr_vel < self.target_velocity + 3:
            reward_mixed = 1 * (1 - d_norm) * alpha_rew

        if curr_vel >= self.target_velocity + 3:
            reward_mixed = (1 - ((curr_vel - self.target_velocity) / (self.max_velocity - self.target_velocity))) * (
                    1 - d_norm) * alpha_rew

        if curr_vel >= self.max_velocity:  # over-speed
            reward_mixed = -10  # (self.max_velocity - curr_vel) * 3.6
            done = True
            result = "over-speed"

        if collision:
            done = True
            reward_collision = -50
            result = "collision"

        if lane == 2 and step < 8: print(f"reset is on Sidewalk line: {lane}")
        if lane == 1:
            # result = "Cross center line"
            pass
        elif lane == 2:
            # done = True # we cancel the cross sidewalk line terminate condition.
            reward_sidewalk = -10
            result = "Cross Sidewalk line"

        if curr_dis_lateral > self.run_off_dis_threshold:
            done = True
            result = "run-off"

        R = reward_collision + reward_sidewalk + reward_mixed + reward_invalid_stop + reward_smooth + reward_deviation

        # ego_vehicle_location = self.ego_vehicle_actor.get_location()

        if len(self.global_planner.global_path_waypoints) == 0:
            done = True
            result = "Goal"

        if step >= 700:
            done = True
            result = "Time out"

        # print(f"****************************** STEP : {step} *************************************")
        # print(f"Delta steer compared to last steer : {np.abs(current_steer - self.previous_steer)}")
        # print(f"Delta angle diff between ego and wpt: {curr_diff_angle}")
        # print(f"Delta lateral distance between ego and wpt: {curr_dis_lateral}")
        # print(f"Reward collision: {reward_collision}")
        # print(f"Reward sidewalk: {reward_sidewalk}")
        # print(f"Reward mixed: {reward_mixed}")
        # print(f"Reward invalid stop: {reward_invalid_stop}")
        # print(f"Reward smooth: {reward_smooth}")
        # print(f"Reward deviation: {reward_deviation}")
        # print(f"**********************************************************************************")


        return R, done, result



    def reward_stcr_risk_aware(self, step):
        """
            Budapest's Thesis Reward Style .
            :param step:
            :return:
        """
        done = False
        result = 0
        reward_run_off = 0
        reward_sidewalk = 0
        reward_collision = 0
        reward_mixed = 0
        reward_invalid_stop = 0
        reward_smooth = 0
        reward_deviation = 0
        _, _, _, propri_obs = self.get_propriceptive_obs()  # acquire propri-obs
        lane = self.get_lane_state()
        collision = self.get_collision_state()

        current_throttle = propri_obs[0]
        current_steer = propri_obs[1]

        if np.abs(current_steer - self.previous_steer) > 0.3:
            reward_smooth = -0.07
        curr_vel = propri_obs[2]  # current vel
        curr_dis_lateral = propri_obs[4]  # current lateral distance to lane
        curr_diff_angle = propri_obs[5]  # current diff angle between wpt and agent

        if curr_dis_lateral >= 0.3: # original 1.05
            reward_deviation = -0.1

        alpha_rew = 1 - np.abs(curr_diff_angle / self.angle_diff_max) if np.abs(
            curr_diff_angle) < self.angle_diff_max else 0
        d_norm = 1 / 3  # 1 / self.run_off_dis_threshold

        # check if there are obstacles on the next 10 target waypoints (around 10 m)
        empty_flag = self.target_lane_empty_check()

        if curr_vel < 0.3 and empty_flag == True:  # and len(self.surrounding_objs) == 0
            self.invalid_stop_frames += 1
        else:
            self.invalid_stop_frames = 0

        if self.invalid_stop_frames > 100:
            done = True
            reward_invalid_stop = -10
            result = "invalid-stop"

        # different reward acquire in different velocity mode.
        if curr_vel < self.min_velocity:
            reward_mixed = curr_vel / self.min_velocity * (1 - d_norm) * alpha_rew * 1.5

        if curr_vel >= self.target_velocity - 3 and curr_vel < self.target_velocity + 3:
            reward_mixed = 1 * (1 - d_norm) * alpha_rew * 3.0

        if curr_vel >= self.target_velocity + 3:
            reward_mixed = (1 - ((curr_vel - self.target_velocity) / (self.max_velocity - self.target_velocity))) * (
                    1 - d_norm) * alpha_rew * 1.5

        if curr_vel >= self.max_velocity:  # over-speed
            reward_mixed = -10  # (self.max_velocity - curr_vel) * 3.6
            done = True
            result = "over-speed"

        if collision:
            done = True
            reward_collision = -10
            result = "collision"

        if lane == 2 and step < 8: print(f"reset is on Sidewalk line: {lane}")
        if lane == 1:
            # result = "Cross center line"
            pass
        elif lane == 2:
            done = True # we cancel the cross sidewalk line terminate condition.
            reward_sidewalk = -25
            result = "Cross Sidewalk line"

        if curr_dis_lateral > self.run_off_dis_threshold:
            done = True
            reward_run_off = -25
            result = "run-off"

        R = reward_collision + reward_sidewalk + reward_mixed + reward_invalid_stop + reward_smooth + reward_deviation + reward_run_off

        # ego_vehicle_location = self.ego_vehicle_actor.get_location()

        if len(self.global_planner.global_path_waypoints) == 0:
            done = True
            result = "Goal"

        if step >= 500: # 550
            done = True
            result = "Time out"

        # Risk Aware Index
        temporal_risk_index = 1 if self.temporal_risk_aware else 0
        lam1 = 0.5
        lam2 = 0.7
        lam3 = 0.4
        risk_aware_reward = curr_vel * (lam1 * (1 - self.cad_ranges[191] / 15) + lam2 * (1 - np.cos(math.radians(curr_diff_angle)))) + lam3 * temporal_risk_index

        R = R - risk_aware_reward


        # print(f"****************************** STEP : {step} *************************************")
        # print(f"Delta steer compared to last steer : {np.abs(current_steer - self.previous_steer)}")
        # print(f"Delta angle diff between ego and wpt: {curr_diff_angle}")
        # print(f"Delta lateral distance between ego and wpt: {curr_dis_lateral}")
        # print(f"Reward collision: {reward_collision}")
        # print(f"Reward sidewalk: {reward_sidewalk}")
        # print(f"Reward mixed: {reward_mixed}")
        # print(f"Reward invalid stop: {reward_invalid_stop}")
        # print(f"Reward smooth: {reward_smooth}")
        # print(f"Reward deviation: {reward_deviation}")

        # print(f"Risk spatial: {curr_vel * lam1 * (1 - self.cad_ranges[191]/15)}")
        # print(f"Risk run-off: {curr_vel * lam2 * (1 - np.cos(math.radians(curr_diff_angle)))}")
        # print(f"Risk temporal: {lam3 * temporal_risk_index}")

        # print(f"Total Reward: {R}")
        # print(f"**********************************************************************************")

        return R, done, result



    def test_reward(self, step):
        """
        This reward function is designed for test episode. The terminate condition is:
        1. run off the road .
        2. goal arrive .
        3. invalid stop .

        :param step: current step
        :return:
            R: reward of current step.
            done: terminate or not.
            result: current step result.
            curr_dis_lateral: lane-off distance of current step. ALOD metric.
            is_collision_step: collision or not of current step.

        """

        is_collision_step = False
        done = False
        result = 0
        reward_sidewalk = 0
        reward_collision = 0
        reward_mixed = 0
        reward_invalid_stop = 0
        reward_smooth = 0
        reward_deviation = 0

        av_location, _, _, propri_obs = self.get_propriceptive_obs()
        lane = self.get_lane_state()
        collision = self.get_collision_state()
        current_throttle = propri_obs[0]
        current_steer = propri_obs[1]

        # reward smooth
        if np.abs(current_steer - self.previous_steer) > 0.3:
            reward_smooth = -0.07

        curr_vel = propri_obs[2]
        curr_dis_lateral = propri_obs[4]
        curr_diff_angle = propri_obs[5]

        if curr_dis_lateral >= 1.05:
            reward_deviation = -0.1

        # reward invalid stop
        empty_flag = self.target_lane_empty_check()

        if curr_vel < 0.3 and empty_flag == True:  # and len(self.surrounding_objs) == 0
            self.invalid_stop_frames += 1
        else:
            self.invalid_stop_frames = 0

        if self.invalid_stop_frames > 100:
            done = True
            reward_invalid_stop = -10
            result = "invalid-stop"


        # reward mixed. different velocity .
        alpha_rew = 1 - np.abs(curr_diff_angle / self.angle_diff_max) if np.abs(
            curr_diff_angle) < self.angle_diff_max else 0
        d_norm = 1 / 3  # 1 / self.run_off_dis_threshold

        # different reward acquire in different velocity mode.
        if curr_vel < self.min_velocity:
            reward_mixed = curr_vel / self.min_velocity * (1 - d_norm) * alpha_rew

        if curr_vel >= self.target_velocity - 3 and curr_vel < self.target_velocity + 3:
            reward_mixed = 1 * (1 - d_norm) * alpha_rew

        if curr_vel >= self.target_velocity + 3:
            reward_mixed = (1 - ((curr_vel - self.target_velocity) / (self.max_velocity - self.target_velocity))) * (
                    1 - d_norm) * alpha_rew

        if curr_vel >= self.max_velocity:  # over-speed
            reward_mixed = -10  # (self.max_velocity - curr_vel) * 3.6
            result = "over-speed"

        # reward collision
        if collision and self.collision_with_actor not in self.collision_victims:
            # If the ego vehicle collide with a new actor, add the victims into the Record List.
            self.collision_victims.append(self.collision_with_actor)
            reward_collision = -10
            result = "collision"

        # reward lane invasion
        if lane == 2 and step < 8: print(f"reset is on Sidewalk line: {lane}")
        if lane == 1:
            # result = "Cross center line"
            pass
        elif lane == 2:
            # done = True
            reward_sidewalk = -10
            result = "Cross Sidewalk line"

        # totally reward .
        R = reward_collision + reward_sidewalk + reward_mixed + reward_invalid_stop + reward_smooth + reward_deviation


        # terminate condition of test episode.
        # case I : if self-vehicle run off the road, terminate test episode.
        if curr_dis_lateral > self.run_off_dis_threshold:
            done = True
            result = "run-off"
        # case II: if self-vehicle finish the entire global plan, terminate test episode.
        if len(self.global_planner.global_path_waypoints) <= 1:
            done = True
            result = "Goal Arrived"
        # case III: if the self-vehicle meet `invalid-stop`, also terminate test episode.


        if done == True:
            self.run_step([0, -1])


        return R, done, result, curr_dis_lateral, is_collision_step



    def test_reward_adversarial(self, step):
        """
        This is designed for test adversarial scenarios events.
        """
        is_collision_step = False
        done = False
        result = 0
        reward_sidewalk = 0
        reward_collision = 0
        reward_mixed = 0
        reward_invalid_stop = 0
        reward_smooth = 0
        reward_deviation = 0

        av_location, _, _, propri_obs = self.get_propriceptive_obs()
        lane = self.get_lane_state()
        collision = self.get_collision_state()
        current_throttle = propri_obs[0]
        current_steer = propri_obs[1]

        # reward smooth
        if np.abs(current_steer - self.previous_steer) > 0.3:
            reward_smooth = -0.07

        curr_vel = propri_obs[2]
        curr_dis_lateral = propri_obs[4]
        curr_diff_angle = propri_obs[5]

        if curr_dis_lateral >= 1.05:
            reward_deviation = -0.1

        # reward invalid stop
        # empty_flag = self.target_lane_empty_check()

        if curr_vel < 0.3: # and empty_flag == True:  # and len(self.surrounding_objs) == 0
            self.invalid_stop_frames += 1
        else:
            self.invalid_stop_frames = 0


        if self.invalid_stop_frames > 100:
            done = True
            reward_invalid_stop = -10
            result = "invalid-stop"


        # reward mixed. different velocity .
        alpha_rew = 1 - np.abs(curr_diff_angle / self.angle_diff_max) if np.abs(
            curr_diff_angle) < self.angle_diff_max else 0
        d_norm = 1 / 3  # 1 / self.run_off_dis_threshold

        # different reward acquire in different velocity mode.
        if curr_vel < self.min_velocity:
            reward_mixed = curr_vel / self.min_velocity * (1 - d_norm) * alpha_rew

        if curr_vel >= self.target_velocity - 3 and curr_vel < self.target_velocity + 3:
            reward_mixed = 1 * (1 - d_norm) * alpha_rew

        if curr_vel >= self.target_velocity + 3:
            reward_mixed = (1 - ((curr_vel - self.target_velocity) / (self.max_velocity - self.target_velocity))) * (
                    1 - d_norm) * alpha_rew

        if curr_vel >= self.max_velocity:  # over-speed
            reward_mixed = -10  # (self.max_velocity - curr_vel) * 3.6
            result = "over-speed"

        # reward collision
        if collision and self.collision_with_actor not in self.collision_victims:
            # If the ego vehicle collide with a new actor, add the victims into the Record List.
            self.collision_victims.append(self.collision_with_actor)
            reward_collision = -10
            result = "collision"

        # reward lane invasion
        if lane == 2 and step < 8: print(f"reset is on Sidewalk line: {lane}")
        if lane == 1:
            # result = "Cross center line"
            pass
        elif lane == 2:
            # done = True
            reward_sidewalk = -10
            result = "Cross Sidewalk line"

        # totally reward .
        R = reward_collision + reward_sidewalk + reward_mixed + reward_invalid_stop + reward_smooth + reward_deviation

        # terminate condition of test episode.
        # case I : if self-vehicle run off the road, terminate test episode.
        if curr_dis_lateral > self.run_off_dis_threshold:
            done = True
            result = "run-off"
        # case II: if self-vehicle finish the entire global plan, terminate test episode.
        if len(self.global_planner.global_path_waypoints) <= 1:
            done = True
            result = "Goal Arrived"
        # case III: if the self-vehicle meet `invalid-stop`, also terminate test episode.


        if done == True:
            self.run_step([0, -1])


        return R, done, result, curr_dis_lateral, is_collision_step


    def reset(self):
        self.run_step([0.0, -0.85])

        # random choice a scenarios.
        scenarios = random.choice(list(scenarios_route.keys()))
        self.scenarios_route = scenarios_route[scenarios]
        self.place_ego_vehicle_api()
        self.send_goal()

        global_path_msg = rospy.wait_for_message("/carla/ego_vehicle/waypoints", Path)
        rospy.loginfo("Global Waypoints GOT !")
        self.global_planner = GlobalPlanner(global_path_msg)
        self.global_planner.generate_global_path_waypoints()
        self.next_wpt = self.global_planner.global_path_waypoints.popleft()  # whenever do reset, popleft the first wpt.
        # self.place_ego_vehicle()

        rospy.loginfo("*************** Start Pose ! ***************")
        # _, _, _, speed = self.get_propriceptive_obs()
        while True:
            self.run_step([0.0, 0])
            # self.place_ego_vehicle()
            self.place_ego_vehicle_api()
            ego_location, ego_heading, _, speed = self.get_propriceptive_obs()
            if speed[2] <= 1 and np.abs(ego_location[0]-self.scenarios_route[0][0])< 0.3 and np.abs(ego_location[1] - self.scenarios_route[0][1]) < 0.3:
                break
        rospy.sleep(0.5)
        self.collision_with_actor = None
        self.collision_victims = []
        self.crossed_lane_markings = None
        self.surrounding_objs = []
        self.invalid_stop_frames = 0

        # Risk Aware
        self.temporal_risk_aware = None

        self.previous_steer = 0 # record last steer.
        while self.ego_odom is None or self.ego_vehicle_status is None or self.next_wpt is None:
            print(f"Waiting for Reset ... ")
        rospy.loginfo("Achieve Reset !")


    def reset_test(self, test_scenarios):
        self.run_step([0.0, -0.85])

        # random choice a scenarios.
            # scenarios = random.choice(list(scenarios_route.keys()))
        scenarios = test_scenarios
        self.scenarios_route = scenarios_route[scenarios]

        self.place_ego_vehicle_api()
        self.send_goal()

        global_path_msg = rospy.wait_for_message("/carla/ego_vehicle/waypoints", Path)
        rospy.loginfo("Global Waypoints GOT !")
        self.global_planner = GlobalPlanner(global_path_msg)
        self.global_planner.generate_global_path_waypoints()

        self.total_length = len(self.global_planner.global_path_waypoints)

        self.next_wpt = self.global_planner.global_path_waypoints.popleft()  # whenever do reset, popleft the first wpt.
        # self.place_ego_vehicle()

        rospy.loginfo("*************** Start Pose ! ***************")
        # _, _, _, speed = self.get_propriceptive_obs()
        while True:
            self.run_step([0.0, 0])
            # self.place_ego_vehicle()
            self.place_ego_vehicle_api()
            ego_location, ego_heading, _, speed = self.get_propriceptive_obs()
            if speed[2] <= 1 and np.abs(ego_location[0]-self.scenarios_route[0][0])< 0.3 and np.abs(ego_location[1] - self.scenarios_route[0][1]) < 0.3:
                break
        rospy.sleep(0.5)
        self.collision_with_actor = None
        self.collision_victims = []
        self.crossed_lane_markings = None
        self.surrounding_objs = []
        self.invalid_stop_frames = 0

        # velocity deque
        # self.view_img = None
        # self.vel_fig, self.vel_ax = plt.subplots()

        self.history_velocity = collections.deque(maxlen= 100)

        self.previous_steer = 0 # record last steer.
        while self.ego_odom is None or self.ego_vehicle_status is None or self.next_wpt is None:
            print(f"Waiting for Reset ... ")
        rospy.loginfo("Achieve Reset !")



    def affected_by_traffic_light(self, lights_list=None, max_distance=None):
        """
        Method to check if there is a red light affecting the vehicle.

            :param lights_list (list of carla.TrafficLight): list containing TrafficLight objects.
                If None, all traffic lights in the scene are used
            :param max_distance (float): max distance for traffic lights to be considered relevant.
                If None, the base threshold value is used
        """

        if not lights_list:
            lights_list = self.world.get_actors().filter("*traffic_light*")

        if not max_distance:
            max_distance = 1 # self._base_tlight_threshold

        if self._last_traffic_light:
            if self._last_traffic_light.state != carla.TrafficLightState.Red:
                self._last_traffic_light = None
            else:
                return (True, self._last_traffic_light)

        ego_vehicle_location = self.ego_vehicle_actor.get_location()
        ego_vehicle_waypoint = self.map.get_waypoint(ego_vehicle_location)

        for traffic_light in lights_list:
            object_location = get_trafficlight_trigger_location(traffic_light)
            object_waypoint = self.map.get_waypoint(object_location)

            if object_waypoint.road_id != ego_vehicle_waypoint.road_id:
                continue

            ve_dir = ego_vehicle_waypoint.transform.get_forward_vector()
            wp_dir = object_waypoint.transform.get_forward_vector()
            dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

            if dot_ve_wp < 0:
                continue

            if traffic_light.state != carla.TrafficLightState.Red:
                continue

            if is_within_distance(object_waypoint.transform, self.ego_vehicle_actor.get_transform(), max_distance, [0, 90]):
                self._last_traffic_light = traffic_light
                return (True, traffic_light)

        return (False, None)


    def get_traffic_light_state(self):
        """
        Return:
            False: no red light ;
            True: meet the red light ;
        """
        actor_list = self.world.get_actors()
        lights_list = actor_list.filter("*traffic_light")
        affected, _ = self.affected_by_traffic_light(lights_list)
        return affected



if __name__ == "__main__":
    bev_range = [-10, -20, 30, 20]         # [x_low, y_low, x_high, y_high]
    bev_img_size = [512, 512]              # [H, W], temporally make sure H == W !
    ego_vehicle_dimensions = [2.52, 1.47]  # [length, width] in m.
    target_velocity = 5  # m / s
    bev_env = BEVENV(bev_range, bev_img_size, ego_vehicle_dimensions, target_velocity)
    action = [0, 0.5]
    action_brake = [0, 0.5]

    for episode in range(1000):
        step = 0
        terminate = False
        result = None
        bev_env.reset()
        while not rospy.is_shutdown() and not terminate:
            act = action if step < 90 else action_brake
            bev_env.run_step(act)
            location, _, _, propri_obs = bev_env.get_propriceptive_obs()

            print(f"current throttle: {propri_obs[0]}")
            print(f"current steer: {propri_obs[1]}")
            print(f"current linear vel: {propri_obs[2]}")
            print(f"current angular vel: {propri_obs[3]}")
            print(f"current run-off dis: {propri_obs[4]}")
            print(f"current orientation diff-angle: {propri_obs[5]}")
            # reward, terminate, result = bev_env.reward_mark_XIV(step)
            reward, terminate, result = bev_env.reward_stcr_risk_aware(step)
            bev_env.render()
            step += 1
            # bev_env.get_route_bev()
            # bev_env.get_cad_bev()
            # bev_env.get_detection_bev()
            # bev_env.get_dynamic_bev()

        print(f"Episode: {episode}, terminate: {terminate}, result: {result}")