cmake_minimum_required(VERSION 3.0.2)
project(cad_gt_carla)


find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  message_filters
  roscpp
  std_msgs
  tf2_ros
)
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)


catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES cad_gt_carla
#  CATKIN_DEPENDS cv_bridge image_transport message_filters roscpp std_msgs tf2_ros
#  DEPENDS system_lib
)


include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

link_libraries(
        ${catkin_LIBRARIES}
        ${PCL_LIBRARIES}
)

add_executable(cad_gt_carla src/cad_gt_carla.cpp)