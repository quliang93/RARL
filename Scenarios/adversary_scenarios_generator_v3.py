"""
ASG_V2, Designed by ZYN, Date: 2024-4-4

In this version, we generate adversarial event from global pose and convert it into CARLA's world coordinates.
Furthermore, we also make the Actor random, which means the generated actor could be a vehicle, motorbike, pedestrian etc.

Notice I: merge start point should be more advanced !
          crossing start point should be more advanced !
"""

import math
import json
import sys
import time
import weakref
import rospy
import carla
import random
import numpy as np
from carla import Vector3D
from collections import deque
import tf.transformations as tf_trans
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseArray, Pose
from carla_msgs.msg import CarlaEgoVehicleInfo

# "vehicle.carlamotors.firetruck" "vehicle.micro.microlino","vehicle.tesla.model3"
# vehicle_blueprints_default = ["vehicle.audi.a2", "vehicle.lincoln.mkz_2017", "vehicle.dodge.charger_police", ]
vehicle_blueprints_default = ["vehicle.dodge.charger_police", "vehicle.audi.a2", "vehicle.tesla.model3"]

bicycle_blueprints_default = ["vehicle.kawasaki.ninja"] # "vehicle.bh.crossbike",

# to record all spawned adversarial events .
spawned_adversarial_events = []


def record_spawned_events(scenarios_name: str, adversarial_type, results):
    with open(f"ASG_spawned/{scenarios_name}_{adversarial_type}.json", "w", encoding= 'utf-8') as ff:
        json.dump(results, ff, ensure_ascii=False, indent=4)
    print(f"Quit !")


class AdversaryScenariosGenerator:
    def __init__(self, adversarial_type):
        self.adversarial_type = adversarial_type # chose one adversarial type
        self.adversarial_actors = [] # all generated adversarial actors.
        self.ego_vehicle_pose = None
        client = carla.Client("127.0.0.1", 2000)
        client.set_timeout(2000)
        self.world = client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        rospy.loginfo(f"CARLA CONNECTED!")

        # Acquire the ambush Route for self-agent
        ego_global_plan_msg = rospy.wait_for_message("/carla/ego_vehicle/waypoints", Path, timeout= 1)
        self.ego_global_path = ego_global_plan_msg.poses # we pick out the ego vehicle's global plan (poses)

        ego_vehicle_msg = rospy.wait_for_message("/carla/ego_vehicle/vehicle_info", CarlaEgoVehicleInfo, timeout= 1)
        self.ego_vehicle_actor = self.world.get_actor(ego_vehicle_msg.id) # acquire ego vehicle actor
        self.ambush_poses_pub = rospy.Publisher("/spawn_poses", PoseArray, queue_size= 1)

        odom_sub = rospy.Subscriber("/carla/ego_vehicle/odometry", Odometry, self.odom_callback)

        self.current_actor = None



    def odom_callback(self, odom_msg: Odometry):
        """
        We record self-agent's odometry information, [x, y, z, qx, qy, qz, qw]
        :param odom_msg:
        :return: None
        """
        self.ego_vehicle_pose = [odom_msg.pose.pose.position.x, odom_msg.pose.pose.position.y, odom_msg.pose.pose.position.z,
                                 odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y, odom_msg.pose.pose.orientation.z, odom_msg.pose.pose.orientation.w]

        self.ego_vehicle_velocity = odom_msg.twist.twist.linear.x # m/s


    def set_crossing_ambush_events(self):
        """
        Set Crossing Ambush Events.
        :return: a deque of crossing events.
        """
        crossing_ambush_events = []
        global_path_length = len(self.ego_global_path)

        for i in range(15, global_path_length, 15):
            event = {}
            carlasp_wpt = carla.Transform()
            crossing_bp = self.blueprint_library.find(random.choice(bicycle_blueprints_default))

            pose = self.ego_global_path[i]
            global_wpt = np.array([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z,
                                   pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z,
                                   pose.pose.orientation.w])
            curr_r, curr_p, curr_y = tf_trans.euler_from_quaternion(global_wpt[3:])
            R_wpt2map = np.array([
                [np.cos(curr_y), -np.sin(curr_y)],
                [np.sin(curr_y), np.cos(curr_y)]
            ])

            crossing_to = random.choice(['left', 'right'])
            if crossing_to == 'left': # the crossing agent go from right to left
                diff_point = np.array([[0], [-2.3]]) # -1.8
                event["offset"] = 2.3 # offset distance
                yaw_carla = math.degrees(-curr_y) - 90 # heads to left in carla world
                trigger_distance = 4 # 7
                throttle = 0.2
            else:
                diff_point = np.array([[0], [5.6]]) # 5.6
                event["offset"] = 5.6 # offset distance
                yaw_carla = math.degrees(-curr_y) + 90 # heads to right in carla world
                trigger_distance = 7 # 15
                throttle = 0.4

            diff_spawn_point = global_wpt[:2].reshape(2, 1) + R_wpt2map @ diff_point
            diff_spawn_point = diff_spawn_point.T
            # Do transformation.
            carlasp_wpt.location.x = diff_spawn_point[0][0]
            carlasp_wpt.location.y = -diff_spawn_point[0][1]
            carlasp_wpt.location.z = 0.3
            carlasp_wpt.rotation.yaw = yaw_carla

            event["global_wpt"] = global_wpt
            event["carla_sp_wpt"] = carlasp_wpt
            event["blueprint"] = crossing_bp

            crossing_control = carla.VehicleControl(throttle= throttle, steer=0)
            event["carla_velocity"] = crossing_control
            event["trigger_distance"] = trigger_distance

            crossing_ambush_events.append(event)



        return deque(crossing_ambush_events)

    # @staticmethod
    def on_collision(self, ba_actor, event):
        if ba_actor.is_alive:
            ba_actor.destroy()
        # self.current_actor.destroy()
        print(f"Collision Check & Clear out !")

    def set_merge_ambush_events(self):
        """
        In this function, we want to prepare merge reckless behaviour. But make sure the
        adversarial agent start its pose behind the ego vehicle.
        :return: a deque, record merge ambush events
        """
        merge_ambush_events = []
        global_path_length = len(self.ego_global_path)
        # For reckless merge event, we totally generate it according to global path. That means we generate a merge event
        # according to the global path's pose, just generate it on the left (carla.Town02 & carla.Town01) as the self-vehicle
        # obeys the right-hand traffic rule.
        for i in range(20, global_path_length, 20):
            event = {}

            carlasp_wpt = carla.Transform()
            merge_bp = self.blueprint_library.find(random.choice(vehicle_blueprints_default+["vehicle.kawasaki.ninja"]))

            pose = self.ego_global_path[i] # we pick out the current pose on the global path.
            global_wpt = np.array([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z,
                                   pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z,
                                   pose.pose.orientation.w])
            curr_r, curr_p, curr_y = tf_trans.euler_from_quaternion(global_wpt[3:])

            R_wpt2map = np.array([
                [np.cos(curr_y), -np.sin(curr_y)],
                [np.sin(curr_y), np.cos(curr_y)]
            ])

            diff_point = np.array([[-10], [4]])
            event["offset"] = np.sqrt(diff_point[1][0]**2 + diff_point[0][0]**2)

            rotated_angle = math.degrees(np.arctan(np.abs(diff_point[1][0] / diff_point[0][0])))

            diff_spawn_point = global_wpt[:2].reshape(2, 1) + R_wpt2map @ diff_point
            diff_spawn_point = diff_spawn_point.T

            carlasp_wpt.location.x = diff_spawn_point[0][0]
            carlasp_wpt.location.y = -diff_spawn_point[0][1]
            carlasp_wpt.location.z = 0.5

            # for yaw exchange, in ROS right-hand to CARLA left-hand, you only need a '-'
            # but do not forget to change the rads into degrees.
            yaw_carla = math.degrees(-curr_y)
            carlasp_wpt.rotation.yaw = yaw_carla + rotated_angle * np.sign(diff_point[1][0])

            event["global_wpt"] = global_wpt
            event["carla_sp_wpt"] = carlasp_wpt
            event["blueprint"] = merge_bp

            # actor_merge = self.world.try_spawn_actor(event["blueprint"], event["carla_sp_wpt"])

            merge_control = carla.VehicleControl(throttle=0.5, steer=0)

            event["carla_velocity"] = merge_control
            event["trigger_distance"] = 7 # m that mean when the ego-vehicle enters the global pose in 1m circle, trigger this event.

            merge_ambush_events.append(event)

        return deque(merge_ambush_events)


    def set_reverse_ambush_events(self):
        """
        Set reverse ambush events.
        :return: a deque of reverse ambush events.
        """
        reverse_ambush_events = []
        global_path_length = len(self.ego_global_path)

        for i in range(20, global_path_length, 15):
            event = {}
            carlasp_wpt = carla.Transform()
            reverse_bp = self.blueprint_library.find(random.choice(vehicle_blueprints_default))

            pose = self.ego_global_path[i]
            global_wpt = np.array([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z,
                                   pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z,
                                   pose.pose.orientation.w])
            curr_r, curr_p, curr_y = tf_trans.euler_from_quaternion(global_wpt[3:])
            R_wpt2map = np.array([
                [np.cos(curr_y), -np.sin(curr_y)],
                [np.sin(curr_y), np.cos(curr_y)]
            ])
            diff_point = np.array([[10], [random.choice([1.1, 1.0])]])
            event["offset"] = 20
            diff_spawn_point = global_wpt[:2].reshape(2, 1) + R_wpt2map @ diff_point
            diff_spawn_point = diff_spawn_point.T

            carlasp_wpt.location.x = diff_spawn_point[0][0]
            carlasp_wpt.location.y = -diff_spawn_point[0][1]
            carlasp_wpt.location.z = 0.3

            yaw_carla = math.degrees(-curr_y) + 180
            carlasp_wpt.rotation.yaw = yaw_carla

            event["global_wpt"] = global_wpt
            event["carla_sp_wpt"] = carlasp_wpt
            event["blueprint"] = reverse_bp

            reverse_control = carla.VehicleControl(throttle= random.choice([0.5, 0.8]), steer=0)
            event["carla_velocity"] = reverse_control
            event["trigger_distance"] = 10

            reverse_ambush_events.append(event)

        return deque(reverse_ambush_events)


    def set_stop_ambush_events(self):
        """
        For ambush events, we have design cut-in, reverse events, here we want to design a scenarios where
        the vehicle in front of the self-vehicle suddenly stop. Thus, the self-vehicle should stop or in-accelerate
        in time to avoid collision.
        :return: a deque of emergency events.
        """
        emergency_stop_events = []

        global_path_length = len(self.ego_global_path)

        for i in range(10, global_path_length, 20): # improve the density of adversarial events.
            event = {}
            # We set the carla spawn waypoint and the bp.
            carlasp_wpt = carla.Transform()
            merge_bp = self.blueprint_library.find(random.choice(vehicle_blueprints_default))# + bicycle_blueprints_default

            pose = self.ego_global_path[i]  # we pick out the current pose on the global path.
            global_wpt = np.array([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z,
                                   pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z,
                                   pose.pose.orientation.w])

            curr_r, curr_p, curr_y = tf_trans.euler_from_quaternion(global_wpt[3:])
            R_wpt2map = np.array([
                [np.cos(curr_y), -np.sin(curr_y)],
                [np.sin(curr_y), np.cos(curr_y)]
            ])
            diff_point = np.array([[1], [0]])

            event["offset"] = 1

            diff_spawn_point = global_wpt[:2].reshape(2, 1) + R_wpt2map @ diff_point
            diff_spawn_point = diff_spawn_point.T
            # The spawn point (In CARLA world) xyz coords .
            carlasp_wpt.location.x = diff_spawn_point[0][0]
            carlasp_wpt.location.y = -diff_spawn_point[0][1]
            carlasp_wpt.location.z = 0.8
            # The yaw of generation in CARLA world.
            yaw_carla = math.degrees(-curr_y)
            carlasp_wpt.rotation.yaw = yaw_carla

            event["global_wpt"] = global_wpt
            event["carla_sp_wpt"] = carlasp_wpt
            event["blueprint"] = merge_bp

            emergency_stop_control = carla.VehicleControl(throttle=random.choice([0.18, 0.2]), steer=0)

            event["carla_velocity"] = emergency_stop_control
            event["trigger_distance"] = 4

            emergency_stop_events.append(event)

        return deque(emergency_stop_events)


    def transform_velocity_ros2carla(self, carla_sp_wpt, velocity):
        """
        Transform velocity from ros coordinate system to CARLA frame.
        """
        carla_v_vector = Vector3D()
        curr_r, curr_p, curr_y = math.radians(carla_sp_wpt.rotation.roll), math.radians(carla_sp_wpt.rotation.pitch), math.radians(-carla_sp_wpt.rotation.yaw)
        R_wpt2map = np.array([
            [np.cos(curr_y), -np.sin(curr_y)],
            [np.sin(curr_y), np.cos(curr_y)]
        ])
        v_vector = R_wpt2map @ velocity
        v_vector = v_vector.T
        carla_v_vector.x = v_vector[0][0]
        carla_v_vector.y = -v_vector[0][1]

        return carla_v_vector


    def run(self):
        if self.adversarial_type == "merge":
            events = self.set_merge_ambush_events()

        if self.adversarial_type == "stop":
            events = self.set_stop_ambush_events()

        if self.adversarial_type == "crossing":
            events = self.set_crossing_ambush_events()

        if self.adversarial_type == "reverse":
            events = self.set_reverse_ambush_events()

        # print(f"There are {len(events)} candidate {self.adversarial_type} events !")
        next_ambush = events.popleft()

        while not rospy.is_shutdown() and len(events) > 0:
            if not self.ego_vehicle_pose:
                continue

            trigger_distance = np.linalg.norm(next_ambush["global_wpt"][:2] - np.array(self.ego_vehicle_pose[:2]))
            # print(f"trigger distance to next ambush: {trigger_distance}")
            if trigger_distance < next_ambush["trigger_distance"]:
                # control = next_ambush["carla_velocity"]
                current_actor = self.world.try_spawn_actor(next_ambush["blueprint"], next_ambush["carla_sp_wpt"])
                if current_actor is not None:
                    self.current_actor = current_actor
                    # collision listening
                    # actor_ref = weakref.ref(actor)
                    collision_detector_bp = self.blueprint_library.find("sensor.other.collision")
                    collision_detector_actor = self.world.spawn_actor(collision_detector_bp, carla.Transform(), attach_to = self.current_actor)
                    self.adversarial_actors.append(collision_detector_actor) # add the collision detector into actor list.
                    collision_detector_actor.listen(lambda event: self.on_collision(self.current_actor, event))

                    v_ego = self.ego_vehicle_velocity # m/s
                    offset = next_ambush["offset"] # m
                    trigger = next_ambush["trigger_distance"]
                    v_ba = offset * v_ego / np.abs(trigger - v_ego) if self.adversarial_type != "stop" else 0.5
                    attack_action_ros = np.array([[v_ba], [0]])
                    attack_action_carla = self.transform_velocity_ros2carla(next_ambush["carla_sp_wpt"], attack_action_ros)
                    self.current_actor.set_target_velocity(attack_action_carla)
                    rospy.loginfo(f"attacking action velocity {attack_action_carla}")
                    # actor.apply_control(control)
                    self.adversarial_actors.append(self.current_actor)

                    #### recording adversarial event ! ####
                    # ambush_event_str = {"victim_id": str(actor.id), "blueprint_library": next_ambush["blueprint"].tags,
                    #                     "ambush_site": next_ambush["global_wpt"].tolist(),"offset_dis": next_ambush["offset"],
                    #                     "trigger_dis": next_ambush["trigger_distance"],
                    #                     "victim_vel": v_ba, "ego_vel": v_ego}
                    ambush_event_str = {"victim_id": str(self.current_actor.id), "offset_dis": next_ambush["offset"],
                                        "ambush_site": next_ambush["global_wpt"].tolist(),
                                        "trigger_dis": next_ambush["trigger_distance"],
                                        "victim_vel": v_ba, "ego_vel": v_ego}
                    spawned_adversarial_events.append(ambush_event_str)  # we collect spawned events !

                else:
                    rospy.loginfo(f"Failed to spawn {self.adversarial_type} event !")

                next_ambush = events.popleft()

            # world tick
            self.world.wait_for_tick()



    def destroy_scenarios(self):
        brake = carla.VehicleControl(throttle= 0, steer=0)
        self.ego_vehicle_actor.apply_control(brake)
        print(f"Ego vehicle stop ...")
        for actor in self.adversarial_actors:
            if actor.is_alive:
                actor.destroy()


def main():
    rospy.init_node("asg_node")

    if len(sys.argv) > 1:
        adversarial_type = sys.argv[1] # adversarial type.
        scenarios_name = sys.argv[2]   # scenarios name, including town and route index.
        rospy.loginfo(f"Received Adversarial Type: {adversarial_type}")
    else:
        print(f"No Received Adversarial Type, start generation randomly !")
        scenarios_name = "Test"
        # adversarial_type = random.choice(["crossing", "merge", "reverse", "stop"])
        adversarial_type = "reverse"

    asg = AdversaryScenariosGenerator(adversarial_type)

    try:
        asg.run()
    finally:
        asg.destroy_scenarios()
        print("ALL ambush actors destroyed !")
        record_spawned_events(scenarios_name, adversarial_type, spawned_adversarial_events)
        time.sleep(1.5)
        # rospy.sleep(0.5)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')

