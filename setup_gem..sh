# export ROS_MASTER_URI=http://10.11.226.222:11311
# export ROS_IP=10.11.226.222
export ROS_MASTER_URI=http://10.42.0.1:11311
export ROS_IP=10.42.0.1
unset ROS_HOSTNAME
sudo ip route replace 10.42.0.1/32 dev wlo1 src 10.42.0.252
nordvpn allowlist add subnet 10.42.0.0/24