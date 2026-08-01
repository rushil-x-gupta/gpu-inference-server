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

Open a new WSL session
```
nvcc --version
python3 --version
sudo apt update && sudo apt upgrade
sudo apt install python3.12-dev
sudo apt install python3.12-venv
sudo apt install libpython3.12-dev
sudo apt install build-essential
```

Install `uv`
```
sudo apt install curl
curl --version
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Open a new WSL session
```
which uv
cd <intended working dir's parent dir>
ssh-keygen -C "rushilgupta49@gmail.com"
cat ~/.ssh/id_ed25519.pub
git clone git@github.com:rushil-x-gupta/gpu-inference-server.git
cd gpu-inference-server/
uv venv vllm-env --python 3.12 --seed
source vllm-env/bin/activate
uv pip install vllm --torch-backend=cu126
```

Reference:
- [Install and Run Locally LLMs using vLLM library on Windows](https://www.youtube.com/watch?v=APUDwZvcdYs&t=337s)
- [Install vLLM on RTX 5060 Ti (16GB) & RTX 5070 / 5080 / 5090 GPUs | Complete Guide](https://www.youtube.com/watch?v=bwaA29Sf0ME)
