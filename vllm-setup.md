# WSL Setup

In Powershell
```
wsl --list --online
```
Verify that `Ubuntu 24.04` appears amongst the output.

```
wsl --install -d Ubuntu-24.04
```
Install `Ubuntu 24.04`.

```
wsl --list --verbose
```
Verify installation.

# CUDA setup

In WSL
```
mkdir -p developer/github.com/rushil-x-gupta
ls
python --version
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt-get -y install cuda-toolkit-12-6
sudo nano .bashrc
```
In `.bashrc`, add the following to the very bottom of the file.
```
export PATH=/usr/local/cuda-12.6/bin${PATH:+:${PATH}}
```
