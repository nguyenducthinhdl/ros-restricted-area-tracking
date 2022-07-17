FROM nvidia/cuda:11.7.0-devel-ubuntu18.04

# ROS core
ENV TZ=Asia/Kolkata \
    DEBIAN_FRONTEND=noninteractive
RUN apt-get update -q && \
    apt-get upgrade -yq && \
    apt-get install -yq wget curl git build-essential vim sudo lsb-release locales bash-completion
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu bionic main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -k https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | sudo apt-key add -
RUN apt-get update -q && \
    apt-get install -y ros-melodic-desktop-full python-rosdep &&\
    apt-get install -y python-rosinstall python-rosinstall-generator python-wstool build-essential python-catkin-tools python3-vcstool &&\
    rm -rf /var/lib/apt/lists/*
RUN rosdep init
RUN locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8 LANGUAGE=en_US:en LC_ALL=en_US.UTF-8
RUN rosdep update
RUN apt-get update

# ros-melodic-web-video-server has been integrated with WebRTC protocol
RUN apt-get install -y ros-melodic-rosbridge-suite ros-melodic-web-video-server ros-melodic-video-stream-opencv ros-melodic-webrtc-ros
# RUN apt install -y nvidia-cuda-toolkit

ENV ROS_DISTRO melodic
RUN mkdir /data
RUN mkdir /config
RUN mkdir /launch
RUN mkdir /script
RUN mkdir /catkin_workspace
RUN mkdir /catkin_workspace/src
COPY ./data/ /data
COPY ./config/ /config
COPY ./launch/ /launch
COPY ./script/ /script
COPY ./src/ /catkin_workspace/src

EXPOSE 8080
ENV LD_LIBRARY_PATH = $LD_LIBRARY_PATH:/usr/local/cuda/lib64


# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

COPY ./ros_entrypoint.sh /
RUN chmod +x "/ros_entrypoint.sh"
ENTRYPOINT ["/ros_entrypoint.sh"]

WORKDIR /catkin_workspace
RUN chmod -R +x /script/

CMD "/script/runtime_script.bash"

