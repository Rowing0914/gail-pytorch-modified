source .setup-mocap
cd ../gail-pytorch-modified
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin
conda activate il
nvidia-smi
python cleaner.py "train_rl.py"

# dev cmds
CUDA_VISIBLE_DEVICES=0 python train_rl.py --env_name=HalfCheetah-v3 --seed=1
python train_rollout.py
python train_il.py

# test cmds
CUDA_VISIBLE_DEVICES=0 python train_rl.py --env_name=HalfCheetah-v3 --seed=1 --wandb --if_video --eval_freq=100 --video_freq=1

# # === benchmark
# CUDA_VISIBLE_DEVICES=0 nohup python train_rl.py --policy_name=sac --env_name=HalfCheetah-v3 --seed=1 --wandb --wb_group=HalfCheetah-v3-sac-1201 --if_save_weight >&/dev/null &
# CUDA_VISIBLE_DEVICES=1 nohup python train_rl.py --policy_name=sac --env_name=HalfCheetah-v3 --seed=2 --wandb --wb_group=HalfCheetah-v3-sac-1201 >&/dev/null &
# CUDA_VISIBLE_DEVICES=2 nohup python train_rl.py --policy_name=sac --env_name=HalfCheetah-v3 --seed=3 --wandb --wb_group=HalfCheetah-v3-sac-1201 >&/dev/null &
# CUDA_VISIBLE_DEVICES=0 nohup python train_rl.py --policy_name=sac --env_name=HalfCheetah-v3 --seed=4 --wandb --wb_group=HalfCheetah-v3-sac-1201 >&/dev/null &
# CUDA_VISIBLE_DEVICES=1 nohup python train_rl.py --policy_name=sac --env_name=HalfCheetah-v3 --seed=5 --wandb --wb_group=HalfCheetah-v3-sac-1201 >&/dev/null &
# CUDA_VISIBLE_DEVICES=2 nohup python train_rl.py --policy_name=ppo --env_name=HalfCheetah-v3 --seed=1 --wandb --wb_group=HalfCheetah-v3-ppo-1201 --if_save_weight >&/dev/null &
# CUDA_VISIBLE_DEVICES=0 nohup python train_rl.py --policy_name=ppo --env_name=HalfCheetah-v3 --seed=2 --wandb --wb_group=HalfCheetah-v3-ppo-1201 >&/dev/null &
# CUDA_VISIBLE_DEVICES=1 nohup python train_rl.py --policy_name=ppo --env_name=HalfCheetah-v3 --seed=3 --wandb --wb_group=HalfCheetah-v3-ppo-1201 >&/dev/null &
# CUDA_VISIBLE_DEVICES=2 nohup python train_rl.py --policy_name=ppo --env_name=HalfCheetah-v3 --seed=4 --wandb --wb_group=HalfCheetah-v3-ppo-1201 >&/dev/null &
# CUDA_VISIBLE_DEVICES=0 nohup python train_rl.py --policy_name=ppo --env_name=HalfCheetah-v3 --seed=5 --wandb --wb_group=HalfCheetah-v3-ppo-1201 >&/dev/null &

# CUDA_VISIBLE_DEVICES=1 nohup python train_rl.py --policy_name=sac --env_name=Humanoid-v3 --seed=1 --wandb --wb_group=Humanoid-v3-sac-1201 --if_save_weight >&/dev/null &
# CUDA_VISIBLE_DEVICES=2 nohup python train_rl.py --policy_name=sac --env_name=Humanoid-v3 --seed=2 --wandb --wb_group=Humanoid-v3-sac-1201 >&/dev/null &
# CUDA_VISIBLE_DEVICES=0 nohup python train_rl.py --policy_name=sac --env_name=Humanoid-v3 --seed=3 --wandb --wb_group=Humanoid-v3-sac-1201 >&/dev/null &
# CUDA_VISIBLE_DEVICES=1 nohup python train_rl.py --policy_name=sac --env_name=Humanoid-v3 --seed=4 --wandb --wb_group=Humanoid-v3-sac-1201 >&/dev/null &
# CUDA_VISIBLE_DEVICES=2 nohup python train_rl.py --policy_name=sac --env_name=Humanoid-v3 --seed=5 --wandb --wb_group=Humanoid-v3-sac-1201 >&/dev/null &
# CUDA_VISIBLE_DEVICES=0 nohup python train_rl.py --policy_name=ppo --env_name=Humanoid-v3 --seed=1 --wandb --wb_group=Humanoid-v3-ppo-1201 --if_save_weight >&/dev/null &
# CUDA_VISIBLE_DEVICES=1 nohup python train_rl.py --policy_name=ppo --env_name=Humanoid-v3 --seed=2 --wandb --wb_group=Humanoid-v3-ppo-1201 >&/dev/null &
# CUDA_VISIBLE_DEVICES=2 nohup python train_rl.py --policy_name=ppo --env_name=Humanoid-v3 --seed=3 --wandb --wb_group=Humanoid-v3-ppo-1201 >&/dev/null &
# CUDA_VISIBLE_DEVICES=0 nohup python train_rl.py --policy_name=ppo --env_name=Humanoid-v3 --seed=4 --wandb --wb_group=Humanoid-v3-ppo-1201 >&/dev/null &
# CUDA_VISIBLE_DEVICES=2 nohup python train_rl.py --policy_name=ppo --env_name=Humanoid-v3 --seed=5 --wandb --wb_group=Humanoid-v3-ppo-1201 >&/dev/null &


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
