# Step 1: Installation
```
bash ./setup_board.sh
```

# Step 2: Export
```
sudo nano ~/.profile

# nvcc
export PATH=${PATH}:/usr/local/cuda/bin
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64
# RealSense
export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.6/pyrealsense2

source ~/.bashrc
sudo reboot
```