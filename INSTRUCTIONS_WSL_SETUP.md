# Running DeepMind Lab on Windows via WSL2

DeepMind Lab **does not support native Windows**. To run it, you must use the Windows Subsystem for Linux (WSL).

## Step 1: Install WSL2
1. Open PowerShell as **Administrator**.
2. Run:
   ```powershell
   wsl --install
   ```
3. Restart your computer if prompted.
4. After restart, an "Ubuntu" terminal window should open automatically.
5. Create a username and password for your Linux system when prompted.

## Step 2: Set up Ubuntu System Dependencies
In your **Ubuntu terminal**, run the following commands. These libraries are **REQUIRED** to build DeepMind Lab.

```bash
# 1. Update package list
sudo apt-get update

# 2. Install dependencies (compiler, python-dev, libraries)
sudo apt-get install -y \
    curl git zip unzip \
    python3-pip python3-dev python3-venv \
    libffi-dev gettext freeglut3-dev libsdl2-dev \
    libosmesa6-dev libglu1-mesa libglu1-mesa-dev \
    libjpeg-dev zlib1g-dev \
    lua5.1 liblua5.1-0-dev \
    gcc g++ build-essential \
    pkg-config software-properties-common
```

## Step 3: Install Miniconda
```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
source ~/.bashrc
```

## Step 4: Install Bazel (Critical for DeepMind Lab)
DeepMind Lab 1.0 does not exist on PyPI. You must build it.
Run these commands in Ubuntu:

```bash
# 1. Install Bazelisk
sudo wget https://github.com/bazelbuild/bazelisk/releases/download/v1.18.0/bazelisk-linux-amd64 -O /usr/local/bin/bazel
sudo chmod +x /usr/local/bin/bazel

# 2. Manually set Bazel version to 5.4.1 (Older versions of DeepMind Lab require Bazel 5)
export USE_BAZEL_VERSION=5.4.1
```

## Step 5: Setup Conda Env & Build DeepMind Lab
We need to remove `deepmind-lab` from `env.yml` first because it fails there.

1. **Edit env.yml**:
   Open `C:\Users\Christian\Documents\GitHub\SHELM\helm\env.yml` in VS Code.
   **Delete** or comment out: `- deepmind-lab==1.0`.

2. **Create Env**:
   ```bash
   cd /mnt/c/Users/Christian/Documents/GitHub/SHELM/helm
   
   PIP_DEFAULT_TIMEOUT=1000 conda env create -f env.yml
   conda activate helm
   ```

3. **Fresh Start DeepMind Lab Build**:
   Inside the `helm` environment:
   ```bash
   cd ~
   rm -rf lab   # Delete old folder to start fresh
   git clone https://github.com/deepmind/lab.git
   cd lab
   
   # Checkout commit from Jan 4 2023 (b1db91a)
   git checkout b1db91af5b4d2f3a24466f4632a3e5e1b0829cca 
   
   # PATCH WORKSPACE FILE:
   # Fix rules_cc URL and SHA (using the correct SHA provided by error message)
   sed -i 's|https://github.com/bazelbuild/rules_cc/archive/master.zip|https://github.com/bazelbuild/rules_cc/archive/40548a2974f1ea06a2f7395e917789a1656G1656.zip|g' WORKSPACE
   sed -i 's|rules_cc-master|rules_cc-40548a2974f1ea06a2f7395e917789a1656G1656|g' WORKSPACE
   sed -i 's|2037875b9a4456dce4a79d114a8af51810cd6e949814127885cb53932766336e|2037875b9a4456dce4a79d112a8ae885bbc4aad968e6587dca6e64f3a0900cdf|g' WORKSPACE

   # Initialize rules_cc toolchains at the END of WORKSPACE
   cat >> WORKSPACE <<EOF
   
   load("@rules_cc//cc:repositories.bzl", "rules_cc_dependencies", "rules_cc_toolchains")
   rules_cc_dependencies()
   rules_cc_toolchains()
   EOF
   
   # Run build command (with Bazel 5.4.1 enforced)
   export USE_BAZEL_VERSION=5.4.1
   export PYTHON_BIN_PATH=$(which python)
   bazel clean --expunge
   bazel build -c opt --python_path=$PYTHON_BIN_PATH //python/pip_package:build_pip_package
   ```
   
   4. **Install Wheel**:
   ```bash
   ./bazel-bin/python/pip_package/build_pip_package /tmp/dmlab_pkg
   pip install /tmp/dmlab_pkg/deepmind_lab-*.whl
   ```

## Step 6: Verify & Run
```bash
# Go back to your project
cd /mnt/c/Users/Christian/Documents/GitHub/SHELM/helm

# Test import
python -c "import deepmind_lab; print('Success!')"

# Run experiment
python main.py --config configs/Psychlab/SHELM/config.json
```
