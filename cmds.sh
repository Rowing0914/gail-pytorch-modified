# Train expert
python train_rl.py --env_name=HalfCheetah-v4 --model_name=trpo
python train_rl.py --env_name=HalfCheetah-v4 --model_name=trpo --wandb --wb_group=trpo
nohup python train_rl.py --env_name=HalfCheetah-v4 --model_name=trpo --wandb >&/dev/null &
python train_rl.py --env_name=HalfCheetah-v4 --model_name=ppo
python train_rl.py --env_name=HalfCheetah-v4 --model_name=ppo --wandb

# train Agent
python train.py --env_name=Pendulum-v1 --wandb