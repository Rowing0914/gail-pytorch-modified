source .setup-mocap
cd ../gail-pytorch-modified
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin
conda activate il
nvidia-smi
python cleaner.py "train_il.py"

# dev cmds
CUDA_VISIBLE_DEVICES=0 python train_rl.py --env_name=HalfCheetah-v3 --seed=1
CUDA_VISIBLE_DEVICES=0 python train_rollout.py --env_name=Humanoid-v3
CUDA_VISIBLE_DEVICES=1 python train_il.py --policy_name=sac --env_name=Humanoid-v3 --seed=1 --eval_freq_by_round=30
CUDA_VISIBLE_DEVICES=1 python train_il.py --policy_name=sac --env_name=HalfCheetah-v3 --seed=1 --eval_freq_by_round=30
CUDA_VISIBLE_DEVICES=1 python train_il.py --policy_name=ppo --env_name=HalfCheetah-v3 --seed=1 --eval_freq_by_round=30
CUDA_VISIBLE_DEVICES=1 python train_il.py --policy_name=sac --env_name=HalfCheetah-v3 --seed=1 --eval_freq_by_round=30
CUDA_VISIBLE_DEVICES=1 python train_il.py --policy_name=sac --env_name=HalfCheetah-v3 --seed=1 --if_no_use_state
CUDA_VISIBLE_DEVICES=1 python train_il.py --policy_name=sac --env_name=HalfCheetah-v3 --seed=1 --if_no_use_action

CUDA_VISIBLE_DEVICES=0 python train_il.py --env_name=HalfCheetah-v3 --seed=1 --eval_freq_by_round=5000 --gen_train_timesteps=8

# # ===== run IL
# ======= Cheetah
# per ts update
# CUDA_VISIBLE_DEVICES=0 nohup python train_il.py --policy_name=sac --env_name=HalfCheetah-v3 --seed=1 --eval_freq_by_round=5000 --gen_train_timesteps=8 --wandb --wb_group=agent-SA-HalfCheetah-v3-sac-1202-noBatch-2 >&/dev/null &
# CUDA_VISIBLE_DEVICES=1 nohup python train_il.py --policy_name=sac --env_name=HalfCheetah-v3 --seed=2 --eval_freq_by_round=5000 --gen_train_timesteps=8 --wandb --wb_group=agent-SA-HalfCheetah-v3-sac-1202-noBatch-2 >&/dev/null &
# CUDA_VISIBLE_DEVICES=2 nohup python train_il.py --policy_name=sac --env_name=HalfCheetah-v3 --seed=3 --eval_freq_by_round=5000 --gen_train_timesteps=8 --wandb --wb_group=agent-SA-HalfCheetah-v3-sac-1202-noBatch-2 >&/dev/null &
# CUDA_VISIBLE_DEVICES=0 nohup python train_il.py --policy_name=sac --env_name=HalfCheetah-v3 --seed=4 --eval_freq_by_round=5000 --gen_train_timesteps=8 --wandb --wb_group=agent-SA-HalfCheetah-v3-sac-1202-noBatch-2 >&/dev/null &

# CUDA_VISIBLE_DEVICES=0 nohup python train_il.py --policy_name=sac --env_name=HalfCheetah-v3 --seed=1 --eval_freq_by_round=5000 --gen_train_timesteps=8 --if_no_use_action --wandb --wb_group=agent-S-HalfCheetah-v3-sac-1202-noBatch-2 >&/dev/null &
# CUDA_VISIBLE_DEVICES=1 nohup python train_il.py --policy_name=sac --env_name=HalfCheetah-v3 --seed=2 --eval_freq_by_round=5000 --gen_train_timesteps=8 --if_no_use_action --wandb --wb_group=agent-S-HalfCheetah-v3-sac-1202-noBatch-2 >&/dev/null &
# CUDA_VISIBLE_DEVICES=2 nohup python train_il.py --policy_name=sac --env_name=HalfCheetah-v3 --seed=3 --eval_freq_by_round=5000 --gen_train_timesteps=8 --if_no_use_action --wandb --wb_group=agent-S-HalfCheetah-v3-sac-1202-noBatch-2 >&/dev/null &
# CUDA_VISIBLE_DEVICES=3 nohup python train_il.py --policy_name=sac --env_name=HalfCheetah-v3 --seed=4 --eval_freq_by_round=5000 --gen_train_timesteps=8 --if_no_use_action --wandb --wb_group=agent-S-HalfCheetah-v3-sac-1202-noBatch-2 >&/dev/null &

# # batch update
# CUDA_VISIBLE_DEVICES=0 nohup python train_il.py --policy_name=sac --env_name=HalfCheetah-v3 --seed=1 --eval_freq_by_round=30 --wandb --wb_group=agent-SA-HalfCheetah-v3-sac-1202-batch-2 >&/dev/null &
# CUDA_VISIBLE_DEVICES=1 nohup python train_il.py --policy_name=sac --env_name=HalfCheetah-v3 --seed=2 --eval_freq_by_round=30 --wandb --wb_group=agent-SA-HalfCheetah-v3-sac-1202-batch-2 >&/dev/null &
# CUDA_VISIBLE_DEVICES=2 nohup python train_il.py --policy_name=sac --env_name=HalfCheetah-v3 --seed=3 --eval_freq_by_round=30 --wandb --wb_group=agent-SA-HalfCheetah-v3-sac-1202-batch-2 >&/dev/null &
# CUDA_VISIBLE_DEVICES=0 nohup python train_il.py --policy_name=sac --env_name=HalfCheetah-v3 --seed=4 --eval_freq_by_round=30 --wandb --wb_group=agent-SA-HalfCheetah-v3-sac-1202-batch-2 >&/dev/null &

# CUDA_VISIBLE_DEVICES=0 nohup python train_il.py --policy_name=sac --env_name=HalfCheetah-v3 --seed=1 --eval_freq_by_round=30 --if_no_use_action --wandb --wb_group=agent-S-HalfCheetah-v3-sac-1202-batch-2 >&/dev/null &
# CUDA_VISIBLE_DEVICES=1 nohup python train_il.py --policy_name=sac --env_name=HalfCheetah-v3 --seed=2 --eval_freq_by_round=30 --if_no_use_action --wandb --wb_group=agent-S-HalfCheetah-v3-sac-1202-batch-2 >&/dev/null &
# CUDA_VISIBLE_DEVICES=2 nohup python train_il.py --policy_name=sac --env_name=HalfCheetah-v3 --seed=3 --eval_freq_by_round=30 --if_no_use_action --wandb --wb_group=agent-S-HalfCheetah-v3-sac-1202-batch-2 >&/dev/null &
# CUDA_VISIBLE_DEVICES=0 nohup python train_il.py --policy_name=sac --env_name=HalfCheetah-v3 --seed=4 --eval_freq_by_round=30 --if_no_use_action --wandb --wb_group=agent-S-HalfCheetah-v3-sac-1202-batch-2 >&/dev/null &

# # =========== PPO
# # per ts update
# CUDA_VISIBLE_DEVICES=0 nohup python train_il.py --policy_name=ppo --env_name=HalfCheetah-v3 --seed=1 --eval_freq_by_round=5000 --gen_train_timesteps=8 --wandb --wb_group=agent-SA-HalfCheetah-v3-ppo-1202-noBatch-2 >&/dev/null &
# CUDA_VISIBLE_DEVICES=1 nohup python train_il.py --policy_name=ppo --env_name=HalfCheetah-v3 --seed=2 --eval_freq_by_round=5000 --gen_train_timesteps=8 --wandb --wb_group=agent-SA-HalfCheetah-v3-ppo-1202-noBatch-2 >&/dev/null &
# CUDA_VISIBLE_DEVICES=2 nohup python train_il.py --policy_name=ppo --env_name=HalfCheetah-v3 --seed=3 --eval_freq_by_round=5000 --gen_train_timesteps=8 --wandb --wb_group=agent-SA-HalfCheetah-v3-ppo-1202-noBatch-2 >&/dev/null &
# CUDA_VISIBLE_DEVICES=0 nohup python train_il.py --policy_name=ppo --env_name=HalfCheetah-v3 --seed=4 --eval_freq_by_round=5000 --gen_train_timesteps=8 --wandb --wb_group=agent-SA-HalfCheetah-v3-ppo-1202-noBatch-2 >&/dev/null &

# CUDA_VISIBLE_DEVICES=0 nohup python train_il.py --policy_name=ppo --env_name=HalfCheetah-v3 --seed=1 --eval_freq_by_round=5000 --gen_train_timesteps=8 --if_no_use_action --wandb --wb_group=agent-S-HalfCheetah-v3-ppo-1202-noBatch-2 >&/dev/null &
# CUDA_VISIBLE_DEVICES=1 nohup python train_il.py --policy_name=ppo --env_name=HalfCheetah-v3 --seed=2 --eval_freq_by_round=5000 --gen_train_timesteps=8 --if_no_use_action --wandb --wb_group=agent-S-HalfCheetah-v3-ppo-1202-noBatch-2 >&/dev/null &
# CUDA_VISIBLE_DEVICES=2 nohup python train_il.py --policy_name=ppo --env_name=HalfCheetah-v3 --seed=3 --eval_freq_by_round=5000 --gen_train_timesteps=8 --if_no_use_action --wandb --wb_group=agent-S-HalfCheetah-v3-ppo-1202-noBatch-2 >&/dev/null &
# CUDA_VISIBLE_DEVICES=0 nohup python train_il.py --policy_name=ppo --env_name=HalfCheetah-v3 --seed=4 --eval_freq_by_round=5000 --gen_train_timesteps=8 --if_no_use_action --wandb --wb_group=agent-S-HalfCheetah-v3-ppo-1202-noBatch-2 >&/dev/null &

# # batch update
# CUDA_VISIBLE_DEVICES=0 nohup python train_il.py --policy_name=ppo --env_name=HalfCheetah-v3 --seed=1 --eval_freq_by_round=30 --wandb --wb_group=agent-SA-HalfCheetah-v3-ppo-1202-batch-2 >&/dev/null &
# CUDA_VISIBLE_DEVICES=1 nohup python train_il.py --policy_name=ppo --env_name=HalfCheetah-v3 --seed=2 --eval_freq_by_round=30 --wandb --wb_group=agent-SA-HalfCheetah-v3-ppo-1202-batch-2 >&/dev/null &
# CUDA_VISIBLE_DEVICES=2 nohup python train_il.py --policy_name=ppo --env_name=HalfCheetah-v3 --seed=3 --eval_freq_by_round=30 --wandb --wb_group=agent-SA-HalfCheetah-v3-ppo-1202-batch-2 >&/dev/null &
# CUDA_VISIBLE_DEVICES=0 nohup python train_il.py --policy_name=ppo --env_name=HalfCheetah-v3 --seed=4 --eval_freq_by_round=30 --wandb --wb_group=agent-SA-HalfCheetah-v3-ppo-1202-batch-2 >&/dev/null &

# CUDA_VISIBLE_DEVICES=0 nohup python train_il.py --policy_name=ppo --env_name=HalfCheetah-v3 --seed=1 --eval_freq_by_round=30 --if_no_use_action --wandb --wb_group=agent-S-HalfCheetah-v3-ppo-1202-batch-2 >&/dev/null &
# CUDA_VISIBLE_DEVICES=1 nohup python train_il.py --policy_name=ppo --env_name=HalfCheetah-v3 --seed=2 --eval_freq_by_round=30 --if_no_use_action --wandb --wb_group=agent-S-HalfCheetah-v3-ppo-1202-batch-2 >&/dev/null &
# CUDA_VISIBLE_DEVICES=2 nohup python train_il.py --policy_name=ppo --env_name=HalfCheetah-v3 --seed=3 --eval_freq_by_round=30 --if_no_use_action --wandb --wb_group=agent-S-HalfCheetah-v3-ppo-1202-batch-2 >&/dev/null &
# CUDA_VISIBLE_DEVICES=0 nohup python train_il.py --policy_name=ppo --env_name=HalfCheetah-v3 --seed=4 --eval_freq_by_round=30 --if_no_use_action --wandb --wb_group=agent-S-HalfCheetah-v3-ppo-1202-batch-2 >&/dev/null &


# ======= Humanoid
# # per ts update
# CUDA_VISIBLE_DEVICES=0 nohup python train_il.py --policy_name=sac --env_name=Humanoid-v3 --seed=1 --eval_freq_by_round=5000 --gen_train_timesteps=8 --wandb --wb_group=agent-SA-Humanoid-v3-sac-1202-noBatch-2 >&/dev/null &
# CUDA_VISIBLE_DEVICES=1 nohup python train_il.py --policy_name=sac --env_name=Humanoid-v3 --seed=2 --eval_freq_by_round=5000 --gen_train_timesteps=8 --wandb --wb_group=agent-SA-Humanoid-v3-sac-1202-noBatch-2 >&/dev/null &
# CUDA_VISIBLE_DEVICES=2 nohup python train_il.py --policy_name=sac --env_name=Humanoid-v3 --seed=3 --eval_freq_by_round=5000 --gen_train_timesteps=8 --wandb --wb_group=agent-SA-Humanoid-v3-sac-1202-noBatch-2 >&/dev/null &
# CUDA_VISIBLE_DEVICES=3 nohup python train_il.py --policy_name=sac --env_name=Humanoid-v3 --seed=4 --eval_freq_by_round=5000 --gen_train_timesteps=8 --wandb --wb_group=agent-SA-Humanoid-v3-sac-1202-noBatch-2 >&/dev/null &

# CUDA_VISIBLE_DEVICES=0 nohup python train_il.py --policy_name=sac --env_name=Humanoid-v3 --seed=1 --eval_freq_by_round=5000 --gen_train_timesteps=8 --if_no_use_action --wandb --wb_group=agent-S-Humanoid-v3-sac-1202-noBatch-2 >&/dev/null &
# CUDA_VISIBLE_DEVICES=1 nohup python train_il.py --policy_name=sac --env_name=Humanoid-v3 --seed=2 --eval_freq_by_round=5000 --gen_train_timesteps=8 --if_no_use_action --wandb --wb_group=agent-S-Humanoid-v3-sac-1202-noBatch-2 >&/dev/null &
# CUDA_VISIBLE_DEVICES=2 nohup python train_il.py --policy_name=sac --env_name=Humanoid-v3 --seed=3 --eval_freq_by_round=5000 --gen_train_timesteps=8 --if_no_use_action --wandb --wb_group=agent-S-Humanoid-v3-sac-1202-noBatch-2 >&/dev/null &
# CUDA_VISIBLE_DEVICES=3 nohup python train_il.py --policy_name=sac --env_name=Humanoid-v3 --seed=4 --eval_freq_by_round=5000 --gen_train_timesteps=8 --if_no_use_action --wandb --wb_group=agent-S-Humanoid-v3-sac-1202-noBatch-2 >&/dev/null &

# batch update
# CUDA_VISIBLE_DEVICES=0 nohup python train_il.py --policy_name=sac --env_name=Humanoid-v3 --seed=1 --eval_freq_by_round=30 --wandb --wb_group=agent-SA-Humanoid-v3-sac-1202-batch-2 >&/dev/null &
# CUDA_VISIBLE_DEVICES=1 nohup python train_il.py --policy_name=sac --env_name=Humanoid-v3 --seed=2 --eval_freq_by_round=30 --wandb --wb_group=agent-SA-Humanoid-v3-sac-1202-batch-2 >&/dev/null &
# CUDA_VISIBLE_DEVICES=2 nohup python train_il.py --policy_name=sac --env_name=Humanoid-v3 --seed=3 --eval_freq_by_round=30 --wandb --wb_group=agent-SA-Humanoid-v3-sac-1202-batch-2 >&/dev/null &
# CUDA_VISIBLE_DEVICES=3 nohup python train_il.py --policy_name=sac --env_name=Humanoid-v3 --seed=4 --eval_freq_by_round=30 --wandb --wb_group=agent-SA-Humanoid-v3-sac-1202-batch-2 >&/dev/null &

# CUDA_VISIBLE_DEVICES=0 nohup python train_il.py --policy_name=sac --env_name=Humanoid-v3 --seed=1 --eval_freq_by_round=30 --if_no_use_action --wandb --wb_group=agent-S-Humanoid-v3-sac-1202-batch-2 >&/dev/null &
# CUDA_VISIBLE_DEVICES=1 nohup python train_il.py --policy_name=sac --env_name=Humanoid-v3 --seed=2 --eval_freq_by_round=30 --if_no_use_action --wandb --wb_group=agent-S-Humanoid-v3-sac-1202-batch-2 >&/dev/null &
# CUDA_VISIBLE_DEVICES=2 nohup python train_il.py --policy_name=sac --env_name=Humanoid-v3 --seed=3 --eval_freq_by_round=30 --if_no_use_action --wandb --wb_group=agent-S-Humanoid-v3-sac-1202-batch-2 >&/dev/null &
# CUDA_VISIBLE_DEVICES=3 nohup python train_il.py --policy_name=sac --env_name=Humanoid-v3 --seed=4 --eval_freq_by_round=30 --if_no_use_action --wandb --wb_group=agent-S-Humanoid-v3-sac-1202-batch-2 >&/dev/null &

# =========== PPO
# per ts update
# CUDA_VISIBLE_DEVICES=0 nohup python train_il.py --policy_name=ppo --env_name=Humanoid-v3 --seed=1 --eval_freq_by_round=5000 --gen_train_timesteps=8 --wandb --wb_group=agent-SA-Humanoid-v3-ppo-1202-noBatch-2 >&/dev/null &
# CUDA_VISIBLE_DEVICES=1 nohup python train_il.py --policy_name=ppo --env_name=Humanoid-v3 --seed=2 --eval_freq_by_round=5000 --gen_train_timesteps=8 --wandb --wb_group=agent-SA-Humanoid-v3-ppo-1202-noBatch-2 >&/dev/null &
# CUDA_VISIBLE_DEVICES=2 nohup python train_il.py --policy_name=ppo --env_name=Humanoid-v3 --seed=3 --eval_freq_by_round=5000 --gen_train_timesteps=8 --wandb --wb_group=agent-SA-Humanoid-v3-ppo-1202-noBatch-2 >&/dev/null &
# CUDA_VISIBLE_DEVICES=3 nohup python train_il.py --policy_name=ppo --env_name=Humanoid-v3 --seed=4 --eval_freq_by_round=5000 --gen_train_timesteps=8 --wandb --wb_group=agent-SA-Humanoid-v3-ppo-1202-noBatch-2 >&/dev/null &

# CUDA_VISIBLE_DEVICES=0 nohup python train_il.py --policy_name=ppo --env_name=Humanoid-v3 --seed=1 --eval_freq_by_round=5000 --gen_train_timesteps=8 --if_no_use_action --wandb --wb_group=agent-S-Humanoid-v3-ppo-1202-noBatch-2 >&/dev/null &
# CUDA_VISIBLE_DEVICES=1 nohup python train_il.py --policy_name=ppo --env_name=Humanoid-v3 --seed=2 --eval_freq_by_round=5000 --gen_train_timesteps=8 --if_no_use_action --wandb --wb_group=agent-S-Humanoid-v3-ppo-1202-noBatch-2 >&/dev/null &
# CUDA_VISIBLE_DEVICES=2 nohup python train_il.py --policy_name=ppo --env_name=Humanoid-v3 --seed=3 --eval_freq_by_round=5000 --gen_train_timesteps=8 --if_no_use_action --wandb --wb_group=agent-S-Humanoid-v3-ppo-1202-noBatch-2 >&/dev/null &
# CUDA_VISIBLE_DEVICES=0 nohup python train_il.py --policy_name=ppo --env_name=Humanoid-v3 --seed=4 --eval_freq_by_round=5000 --gen_train_timesteps=8 --if_no_use_action --wandb --wb_group=agent-S-Humanoid-v3-ppo-1202-noBatch-2 >&/dev/null &

# batch update
# CUDA_VISIBLE_DEVICES=0 nohup python train_il.py --policy_name=ppo --env_name=Humanoid-v3 --seed=1 --eval_freq_by_round=30 --wandb --wb_group=agent-SA-Humanoid-v3-ppo-1202-batch-2 >&/dev/null &
# CUDA_VISIBLE_DEVICES=1 nohup python train_il.py --policy_name=ppo --env_name=Humanoid-v3 --seed=2 --eval_freq_by_round=30 --wandb --wb_group=agent-SA-Humanoid-v3-ppo-1202-batch-2 >&/dev/null &
# CUDA_VISIBLE_DEVICES=1 nohup python train_il.py --policy_name=ppo --env_name=Humanoid-v3 --seed=3 --eval_freq_by_round=30 --wandb --wb_group=agent-SA-Humanoid-v3-ppo-1202-batch-2 >&/dev/null &
# CUDA_VISIBLE_DEVICES=0 nohup python train_il.py --policy_name=ppo --env_name=Humanoid-v3 --seed=4 --eval_freq_by_round=30 --wandb --wb_group=agent-SA-Humanoid-v3-ppo-1202-batch-2 >&/dev/null &

# CUDA_VISIBLE_DEVICES=0 nohup python train_il.py --policy_name=ppo --env_name=Humanoid-v3 --seed=1 --eval_freq_by_round=30 --if_no_use_action --wandb --wb_group=agent-S-Humanoid-v3-ppo-1202-batch-2 >&/dev/null &
# CUDA_VISIBLE_DEVICES=1 nohup python train_il.py --policy_name=ppo --env_name=Humanoid-v3 --seed=2 --eval_freq_by_round=30 --if_no_use_action --wandb --wb_group=agent-S-Humanoid-v3-ppo-1202-batch-2 >&/dev/null &
# CUDA_VISIBLE_DEVICES=2 nohup python train_il.py --policy_name=ppo --env_name=Humanoid-v3 --seed=3 --eval_freq_by_round=30 --if_no_use_action --wandb --wb_group=agent-S-Humanoid-v3-ppo-1202-batch-2 >&/dev/null &
# CUDA_VISIBLE_DEVICES=3 nohup python train_il.py --policy_name=ppo --env_name=Humanoid-v3 --seed=4 --eval_freq_by_round=30 --if_no_use_action --wandb --wb_group=agent-S-Humanoid-v3-ppo-1202-batch-2 >&/dev/null &


# test cmds
CUDA_VISIBLE_DEVICES=0 python train_rl.py --env_name=HalfCheetah-v3 --seed=1 --wandb --if_video --eval_freq=100 --video_freq=1

# # === benchmark RL
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
CUDA_VISIBLE_DEVICES=0 python train_rl.py --policy_name=sac --env_name=Pendulum-v1 --seed=1 --if_save_weight


# train Agent
CUDA_VISIBLE_DEVICES=1 python train.py --env_name=HalfCheetah-v3
python train.py --env_name=Pendulum-v1 --if_obs_only

CUDA_VISIBLE_DEVICES=0 nohup python train.py --env_name=HalfCheetah-v3 --wandb --wb_run=pendulum-obs-action >&/dev/null &
CUDA_VISIBLE_DEVICES=1 nohup python train.py --env_name=HalfCheetah-v3 --if_obs_only --wandb --wb_run=pendulum-obs >&/dev/null &
