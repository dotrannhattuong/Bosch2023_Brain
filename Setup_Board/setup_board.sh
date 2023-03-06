sudo apt update
sudo apt upgrade
sudo apt install python3-pip
sudo apt install python3-pil
sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran libopenblas-dev
pip3 install --upgrade pip
pip3 install --upgrade setuptools
python3 -m pip install aiohttp numpy=='1.19.4' scipy=='1.5.3'
python3 -m pip install --upgrade protobuf

# Jtop 
sudo pip3 install jetson-stats

# Nomachine
wget https://www.nomachine.com/free/arm/v8/deb -O nomachine.deb
sudo dpkg -i nomachine.deb

# nvcc
sudo gedit ~/.bashrc
export PATH=${PATH}:/usr/local/cuda/bin
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64
source ~/.bashrc

# Pytorch
wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl -O torch-1.8.0-cp36-cp36m-linux_aarch64.whl
sudo apt-get install python3-pip libopenblas-base libopenmpi-dev libomp-dev
pip3 install Cython
pip3 install numpy torch-1.8.0-cp36-cp36m-linux_aarch64.whl

# torchvision
sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev
git clone --branch v0.9.0 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.9.0 
python3 setup.py install --user
cd ../

# # Yolov7 Requirements

# Onnx
# pip3 install protobuf==3.19.4
# pip3 install onnxruntime_gpu-1.10.0-cp36-cp36m-linux_aarch64.whl
# sudo apt-get update
# sudo apt-get install python3-pip protobuf-compiler libprotoc-dev
# pip3 install onnx>=1.9.0

# Pycuda 
bash ./install_pycuda.sh

# RealSense
./buildLibrealsense.sh
sudo gedit ~/.bashrc
export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python3.6/pyrealsense2
source ~/.bashrc

