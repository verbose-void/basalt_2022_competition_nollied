# setup java/mineRL
sudo add-apt-repository ppa:openjdk-r/ppa
sudo apt-get update
sudo apt-get install openjdk-8-jdk
pip install git+https://github.com/minerllabs/minerl

# setup my stuff
pip install -e external_libs/vpt/
pip install -e external_libs/fractal-zero/
pip install gym==0.19  # required version for my purposes
