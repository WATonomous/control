# Shared message repo

We have seperate packages for each team right now to avoid naming collisions, since each package puts messages into its own namespace. Here's how make use of a message in one of these packages, using `path_planning_msgs` as an example.

In the package which is using the messages' `package.xml`:
```
<build_depend>path_planning_msgs</build_depend>
...
<run_depend>path_planning_msgs</run_depend>
```

In the package which is using the messages' `CMakeLists.txt`:
```
find_package(...
  path_planning_msgs
...)

generate_messages(...
  DEPENDENCIES
  path_planning_msgs
...)

catkin_package(
  CATKIN_DEPENDS ... path_planning_msgs ...
)
```

## Troubleshooting

1) If you encounter this sort of error when `catkin_make`-ing the `ros_msgs` folder similar to this:
```
-- Could not find the required component 'jsk_recognition_msgs'. The following CMake error indicates that you either need to install the package with the same name or change your environment so that it can be found.
CMake Error at /opt/ros/kinetic/share/catkin/cmake/catkinConfig.cmake:83 (find_package):
  Could not find a package configuration file provided by
  "jsk_recognition_msgs" with any of the following names:

    jsk_recognition_msgsConfig.cmake
    jsk_recognition_msgs-config.cmake

  Add the installation prefix of "jsk_recognition_msgs" to CMAKE_PREFIX_PATH
  or set "jsk_recognition_msgs_DIR" to a directory containing one of the
  above files.  If "jsk_recognition_msgs" provides a separate development
  package or SDK, be sure it has been installed.
Call Stack (most recent call first):
  ros_msgs/path_planning_msgs/CMakeLists.txt:4 (find_package)
```


Try this command:
```
 sudo apt-get install ros-kinetic-jsk-recognition-msgs
```