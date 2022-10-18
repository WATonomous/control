
# ================= Dependencies ===================

# First stage sets up the base, ie everything except the active dev repo
# The carla ROS bridge is only distributed on apt for ubuntu 18 (Bionic)
FROM ros:noetic-ros-core AS base


# Set up apt repo
RUN apt-get update
RUN apt-get install -y lsb-release software-properties-common
RUN apt-get update

# Install catkin build system
RUN apt-get update && apt-get install -y build-essential python3-catkin-tools wget python3-rosdep ros-noetic-tf2-geometry-msgs wget curl python3-rospkg

RUN apt-get install -y python3-pip 
# python-pygame python-tk
RUN pip3 install networkx statistics simple-pid glob2 numpy matplotlib datetime casadi
RUN apt-get install -y ros-noetic-tf
#not sure how to install this
#RUN apt-get install -y python-roslib python-rospy python-tf2 libtf2-ros-dev python-tf2-ros

# Add a docker user so we that created files in the docker container are owned by a non-root user
RUN addgroup --gid 1000 docker && \
    adduser --uid 1000 --ingroup docker --home /home/docker --shell /bin/bash --disabled-password --gecos "" docker && \
    echo "docker ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/nopasswd

# Remap the docker user and group to be the same uid and group as the host user.
# Any created files by the docker container will be owned by the host user.
RUN USER=docker && \
    GROUP=docker && \
    curl -SsL https://github.com/boxboat/fixuid/releases/download/v0.4/fixuid-0.4-linux-amd64.tar.gz | tar -C /usr/local/bin -xzf - && \
    chown root:root /usr/local/bin/fixuid && \
    chmod 4755 /usr/local/bin/fixuid && \
    mkdir -p /etc/fixuid && \
    printf "user: $USER\ngroup: $GROUP\npaths:\n  - /home/docker/" > /etc/fixuid/config.yml

USER docker:docker

ENV DEBIAN_FRONTEND noninteractive
RUN sudo chsh -s /bin/bash
ENV SHELL=/bin/bash

# ================= Repositories ===================

FROM base as repo
# Set up workspace
RUN mkdir -p ~/catkin_ws/src
WORKDIR /home/docker/catkin_ws/src

#copy all the source files from teh monorepo
COPY src/ros_msgs ros_msgs
COPY src/path_planning/feedback feedback

#for package installation in the future
RUN sudo rm /etc/ros/rosdep/sources.list.d/20-default.list || true && \
    sudo rosdep init && \
    sudo apt-get update && \
    rosdep update && \
    rosdep install --from-paths . --ignore-src -r -y

WORKDIR /home/docker/catkin_ws
RUN catkin config --extend /opt/ros/noetic && \
    catkin build && \
    rm -rf .catkin_tools build

# Entrypoint will run before any CMD on launch. Sources ~/opt/<ROS_DISTRO>/setup.bash and ~/catkin_ws/devel/setup.bash
COPY docker/wato_ros_entrypoint.sh /home/docker/wato_ros_entrypoint.sh
RUN sudo chmod +x ~/wato_ros_entrypoint.sh
ENTRYPOINT ["/home/docker/wato_ros_entrypoint.sh"]
CMD ["roslaunch", "--wait", "pp_feedback", "mpc.launch"]