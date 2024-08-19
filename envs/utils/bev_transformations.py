import numpy as np
from nav_msgs.msg import Odometry
from derived_object_msgs.msg import Object
from jsk_recognition_msgs.msg import BoundingBox
from tf import transformations


def yaw_to_rotation_matrix(yaw):
    """
    Transform yaw angle to Rotation matrix 2D()
    :param yaw:
    :return: (2, 2) shape r-matrix
    """
    c = np.cos(yaw)
    s = np.sin(yaw)
    rotation_matrix = np.array([[c, -s], [s, c]])
    return rotation_matrix


def transform_objects_to_ego_vehicle_frame(ego_odom_msg: Odometry, obj: Object) -> (Object, BoundingBox):
    """
    This function could transform object(ROS standard format) from 'map' frame to 'ego_vehicle' frame.
    So, we need to know the object's quaternion and position in ego vehicle frame.
    Param:
        ego_odom_msg: nav_msgs.msg.Odometry
        obj: derived_object_msgs.msg.Object
    Return:
        Object (in ego vehicle frame), notice: here we set the lidar at the center of the ego vehicle.
        BoundingBox (in ego vehicle frame), notice: to display in RViz.

    """
    object_in_lidar = Object()
    rviz_bbox = BoundingBox()
    object_in_lidar.header.frame_id = "ego_vehicle"
    rviz_bbox.header.frame_id = "ego_vehicle"

    # Acquire the rotation matrix from object's quaternion
    obj_rotation_matrix = transformations.quaternion_matrix(
        [obj.pose.orientation.x, obj.pose.orientation.y, obj.pose.orientation.z, obj.pose.orientation.w])

    # we acquire the ego translation matrix in map frame
    ego_translation_matrix = transformations.translation_matrix(
        [ego_odom_msg.pose.pose.position.x, ego_odom_msg.pose.pose.position.y,
         ego_odom_msg.pose.pose.position.z])

    # we acquire the ego rotation matrix , the is ego2map !!!
    ego_rotation_matrix = transformations.quaternion_matrix(
        [ego_odom_msg.pose.pose.orientation.x, ego_odom_msg.pose.pose.orientation.y,
         ego_odom_msg.pose.pose.orientation.z, ego_odom_msg.pose.pose.orientation.w])

    Rotation_obj2map = obj_rotation_matrix # (4, 4)

    Rotation_ego2map = ego_rotation_matrix # (4, 4)

    # Now we know: R_(obj->map), R_(ego->map); we want to calculate R_(obj->ego),
    # that because after we got R_(obj->ego), we could acquire the object's quaternion in ego-vehicle frame !
    # Rotation: (Rotation: ego -> map).inverse @ Rotation: map <- obj
    Rotation_obj2ego = transformations.inverse_matrix(Rotation_ego2map) @ Rotation_obj2map

    obj_quateron_in_ego = transformations.quaternion_from_matrix(
        Rotation_obj2ego)  # Here we got the object's quaternion in ego vehicle frame

    obj_in_map = np.array([obj.pose.position.x, obj.pose.position.y, obj.pose.position.z, 1.0]).reshape(4, 1)
    """
    tf_ego2map = [[R, T]
                  [0, 1]]
    """
    Rotation_ego2map[:, 3] = ego_translation_matrix[:, 3]
    T_ego2map = Rotation_ego2map

    obj_in_ego = np.linalg.inv(T_ego2map) @ obj_in_map

    # Here we can transform object in lidar frame.
    T_ego2lidar = np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 1.70],[0, 0, 0, 1]], dtype= np.float64)

    obj_in_lidar = T_ego2lidar @ obj_in_ego


    # object_in_lidar.shape.dimensions
    object_in_lidar.shape.dimensions = obj.shape.dimensions

    object_in_lidar.classification = obj.classification

    classification_name = {4:"Pedestrian", 5:"Bike", 7:"Truck", 8:"MotorCycle"}

    if obj.classification in classification_name.keys():
        print(f"There is a {classification_name[obj.classification]} nearby !!!")

    # For display in Rviz, I found that we should adjust z a little according to reality
    rviz_bbox.dimensions.x = obj.shape.dimensions[0]
    rviz_bbox.dimensions.y = obj.shape.dimensions[1]
    rviz_bbox.dimensions.z = obj.shape.dimensions[2]

    if obj.classification == 4: # which means Pedestrian class
        object_in_lidar.pose.position.x = obj_in_lidar[0]
        object_in_lidar.pose.position.y = obj_in_lidar[1]
        object_in_lidar.pose.position.z = obj_in_lidar[2] - obj.shape.dimensions[2] * 0.85
        # rviz bbox
        rviz_bbox.pose.position.x = obj_in_lidar[0]
        rviz_bbox.pose.position.y = obj_in_lidar[1]
        rviz_bbox.pose.position.z = obj_in_lidar[2] - obj.shape.dimensions[2] * 0.85

    # For Car class.
    else:
        object_in_lidar.pose.position.x = obj_in_lidar[0]
        object_in_lidar.pose.position.y = obj_in_lidar[1]
        object_in_lidar.pose.position.z = obj_in_lidar[2] - obj.shape.dimensions[2] * 0.4
        # rviz bbox
        rviz_bbox.pose.position.x = obj_in_lidar[0]
        rviz_bbox.pose.position.y = obj_in_lidar[1]
        rviz_bbox.pose.position.z = obj_in_lidar[2] - obj.shape.dimensions[2] * 0.4


    object_in_lidar.pose.orientation.x = obj_quateron_in_ego[0]
    object_in_lidar.pose.orientation.y = obj_quateron_in_ego[1]
    object_in_lidar.pose.orientation.z = obj_quateron_in_ego[2]
    object_in_lidar.pose.orientation.w = obj_quateron_in_ego[3]

    # rviz bbox
    rviz_bbox.pose.orientation.x = obj_quateron_in_ego[0]
    rviz_bbox.pose.orientation.y = obj_quateron_in_ego[1]
    rviz_bbox.pose.orientation.z = obj_quateron_in_ego[2]
    rviz_bbox.pose.orientation.w = obj_quateron_in_ego[3]

    rviz_bbox.label = obj.classification

    return object_in_lidar, rviz_bbox