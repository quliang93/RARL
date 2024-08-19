import numpy as np
from geometry_msgs.msg import Pose, Twist, Accel
from tf.transformations import euler_from_quaternion, quaternion_matrix, rotation_matrix, quaternion_from_euler


class BEVObject:
    """
    This class, BEVObject, is designed for achieving all transformation, prediction, and visualizing.
    Main Methods:
        A function that get the four vertexes of an object in Map frame .
        A function that get the predicted trajectory of an object in Map frame. (Polygon format) .
        A function that get the predicted trajectory of an object in Map frame. (BBoxes format) .
    """
    def __init__(self, pose: Pose, twist: Twist, length, width, cls_id, delta_t=0.1, time_horizon=4): # accel: Accel,
        """
        Initialize BEVObject.
        :param pose:          pose in map
        :param twist:         twist in map
        :param accel:         accel in map
        :param length:        length of this object
        :param width:         width of this object
        :param delta_t:       time interval for predicting
        :param time_horizon:  time horizon for predicting
        """
        self.pose = pose
        self.twist = twist
        # self.accel = accel
        self.length = length
        self.width = width
        self.bbox_h = 0.8
        self.cls_id = cls_id
        if self.cls_id == 4:
            self.width = 1.2  # self.width * 2
            self.length = 1.2 # self.length * 2
        else:
            if self.width < 1.5:
                self.width = 1.5


        self.delta_t = delta_t
        self.time_horizon = time_horizon
        self.total_steps = int(self.time_horizon / self.delta_t)


    def yaw_to_rotation_matrix(self, yaw):
        """
        Transform yaw angle to Rotation matrix 2D()
        :param yaw:
        :return: (2, 2) shape r-matrix
        """
        c = np.cos(yaw)
        s = np.sin(yaw)
        rotation_matrix = np.array([[c, -s], [s, c]])
        return rotation_matrix


    def predict_obj_polygon_vertexes(self):
        """
        This function predict the polygon vertexes of the object given the
        velocity (Twist), dimensions (L, W), and Time horizon with delta .
        :return: np.ndarray, shape (N, 2), N = time_horizon / delta_t * 4 vertexes, 2 means x, y coord
        """
        linear_x = self.twist.linear.x
        linear_y = self.twist.linear.y
        angular_z = self.twist.angular.z
        linear_velocity = np.linalg.norm([linear_x, linear_y])

        if angular_z == 0:
            angular_z += 0.0000001

        _, _, ego_yaw = euler_from_quaternion([
            self.pose.orientation.x, self.pose.orientation.y,
            self.pose.orientation.z, self.pose.orientation.w
        ])

        Rotation_matrix_ego2map = self.yaw_to_rotation_matrix(ego_yaw)

        turning_radius = np.abs(linear_velocity / angular_z)
        delta_yaw = np.array([angular_z * (i + 1)* 0.97 ** (i + 1) * self.delta_t for i in range(int(self.time_horizon / self.delta_t))])
        x_list = turning_radius * np.sin(np.abs(delta_yaw))
        y_list = turning_radius - turning_radius * np.cos(delta_yaw) if angular_z > 0 else -(
                turning_radius - turning_radius * np.cos(delta_yaw))

        local_traj_2D = np.array([x_list, y_list])

        local_traj_map = Rotation_matrix_ego2map @ local_traj_2D  # (2, 8)
        local_traj_map = local_traj_map.T
        ego_pos_point = np.array([self.pose.position.x, self.pose.position.y, self.pose.position.z], dtype= np.float)
        # ego_pos_polygon = self.get_obj_polygon_vertex_in_map()
        # ego_pos_polygon = ego_pos_polygon[2:]
        N = int(self.time_horizon / self.delta_t)

        # we calculate the current Rotation Matrix
        R_homo = quaternion_matrix([self.pose.orientation.x, self.pose.orientation.y,
                                    self.pose.orientation.z, self.pose.orientation.w])
        R = R_homo[:3, :3]
        predicted_polygon_vertexes = []
        # for point in ego_pos_points:
        R_matrix_list = []
        # R_matrix_list.append(R)
        for yaw in delta_yaw:
            q = quaternion_from_euler(0, 0, yaw)
            r_matrix = quaternion_matrix(q)[:3, :3]
            R_temp = r_matrix @ R
            R_matrix_list.append(R_temp)

        for idx, pred_p in enumerate(local_traj_map):
            # yaw = delta_yaw[idx]
            # q = quaternion_from_euler(0, 0, yaw)
            # r_matrix = quaternion_matrix(q)[:3, :3]
            # R = r_matrix @ R

            center_x = pred_p[0] + ego_pos_point[0]
            center_y = pred_p[1] + ego_pos_point[1]
            center_z = ego_pos_point[2]

            # x = [self.length / 2, self.length / 2, -self.length / 2, -self.length / 2]
            # y = [self.width / 2, -self.width / 2, self.width / 2, -self.width / 2]
            # z = [self.bbox_h, self.bbox_h, self.bbox_h, self.bbox_h]

            x = [-self.length / 2, -self.length / 2,  self.length / 2, self.length / 2]
            y = [ self.width / 2,  -self.width / 2,  -self.width / 2,  self.width / 2]
            z = [self.bbox_h, self.bbox_h, self.bbox_h, self.bbox_h]

            center = np.array(
                [center_x, center_y, center_z], dtype=np.float32).reshape(1, 3)
            bbox3D_temp = np.array([x, y, z], dtype=np.float32)
            #
            # R_homo = quaternion_matrix([self.pose.orientation.x, self.pose.orientation.y,
            #                             self.pose.orientation.z, self.pose.orientation.w])
            # R = R_homo[:3, :3]
            Rotated_bbox3D = R_matrix_list[idx] @ bbox3D_temp
            bbox3D_lidar_vertexes = Rotated_bbox3D.T + center

            for item in bbox3D_lidar_vertexes:
                # [x, y, z] hear means the predicted center of the object in future traj.
                # Then, we should calculate the vertexes from [x, y, z] and its [yaw] at step t.
                # predicted_polygon_vertexes.append([item[0], item[1], item[2]])
                predicted_polygon_vertexes.append([item[0], item[1]])

        polygon = np.array(predicted_polygon_vertexes, dtype= np.float)
        indices = self.special_sorted_indices(N)
        polygon_sorted = polygon[indices]

        return polygon_sorted


    def special_sorted_indices(self, N):
        first_edge_indices = []
        second_edge_indices = []
        for idx in range(N):
            first_edge_former = idx * 4
            first_edge_latter = first_edge_former + 3
            first_edge_indices.append(first_edge_former)
            first_edge_indices.append(first_edge_latter)

            second_edge_former = (idx*4)+1
            second_edge_latter = second_edge_former + 1
            second_edge_indices.append(second_edge_former)
            second_edge_indices.append(second_edge_latter)

        second_edge_indices.reverse()
        first_edge_indices.extend(second_edge_indices)
        return first_edge_indices


    def get_four_vertexes_in_map(self):
        """
        This function will get the four vertexes in map frame.
        :return: four vertexes of the object in map frame, shape: (4, 2), np.ndarray
        """
        default_box = np.array([
            [self.length / 2, self.width / 2],
            [self.length / 2, -self.width / 2],
            [-self.length / 2, -self.width / 2],
            [-self.length / 2, self.width / 2]], dtype = np.float32)
        _, _, yaw = euler_from_quaternion([self.pose.orientation.x, self.pose.orientation.y,
                                           self.pose.orientation.z, self.pose.orientation.w])
        rot_matrix = self.yaw_to_rotation_matrix(yaw)
        critical_box_vertexes = rot_matrix @ default_box.T

        critical_box_vertexes[0, :] += self.pose.position.x # add the object's current center x coord.
        critical_box_vertexes[1, :] += self.pose.position.y # add the object's current center y coord.
        return critical_box_vertexes.T
