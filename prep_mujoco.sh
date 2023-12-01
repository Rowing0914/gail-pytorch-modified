if [ ! -d "./.mujoco" ]; then
  echo "./.mujoco does not exist."
  # this doesn't take long
  mkdir -p ./.mujoco &&
    wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz &&
    tar -xf mujoco.tar.gz -C ./.mujoco &&
    rm mujoco.tar.gz
fi

if [ ! -d "/root/.mujoco" ]; then
  echo "$/root/.mujoco does not exist."
  mkdir -p /root/.mujoco
fi

cp -R ./.mujoco/mujoco210 /root/.mujoco/mujoco210

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

echo "start pip install"
pip install mujoco-py==2.1.2.14
echo "done pip install"
