export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin
conda activate il

CUDA_VISIBLE_DEVICES=0 python python train_rl.py
python train_rollout.py
python train_il.py

# === benchmark
CUDA_VISIBLE_DEVICES=0 nohup python python train_rl.py >&/dev/null &

# ============ Old ones
# Train expert
python train_rl.py --env_name=HalfCheetah-v4 --model_name=trpo
python train_rl.py --env_name=HalfCheetah-v4 --model_name=trpo --wandb --wb_group=trpo
nohup python train_rl.py --env_name=HalfCheetah-v4 --model_name=trpo --wandb >&/dev/null &
python train_rl.py --env_name=HalfCheetah-v4 --model_name=ppo
python train_rl.py --env_name=HalfCheetah-v4 --model_name=ppo --wandb

# train Agent
python train.py --env_name=Pendulum-v1
python train.py --env_name=Pendulum-v1 --if_obs_only

CUDA_VISIBLE_DEVICES=0 nohup python train.py --env_name=Pendulum-v1 --wandb --wb_run=pendulum-obs-action >&/dev/null &
CUDA_VISIBLE_DEVICES=0 nohup python train.py --env_name=Pendulum-v1 --if_obs_only --wandb --wb_run=pendulum-obs >&/dev/null &
