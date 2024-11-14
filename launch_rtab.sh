export XAUTH=/tmp/.docker.xauth
touch $XAUTH
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -
sudo systemctl start docker
sudo docker run -it --rm \
   --privileged \
   --env="DISPLAY=$DISPLAY" \
   --env="QT_X11_NO_MITSHM=1" \
   --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
   --env="XAUTHORITY=$XAUTH" \
   --volume="$XAUTH:$XAUTH" \
   -e NVIDIA_VISIBLE_DEVICES=all \
   -e NVIDIA_DRIVER_CAPABILITIES=all \
   -e OMP_WAIT_POLICY=passive \
   --runtime=nvidia \
   --network host \
   -v ~/Documents/RTAB-Map:/root/Documents/RTAB-Map \
   introlab3it/rtabmap:focal \
   rtabmap