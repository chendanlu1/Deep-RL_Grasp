import os
import sys
import time
from loguru import logger

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# === 添加 utils 路径 ===
script_dir = os.path.abspath(os.path.dirname(__file__) if '__file__' in globals() else os.getcwd())
sys.path.append(os.path.join(script_dir, '../utils'))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, CheckpointCallback

from FR_Gym.Fr5_env import FR5_Env
from FR_Gym.Callback import TensorboardCallback
from utils.arguments import get_args

now = time.strftime('%m%d-%H%M%S', time.localtime())
args, kwargs = get_args()

models_dir = args.models_dir
logs_dir = args.logs_dir
checkpoints = args.checkpoints
test = args.test

# 自动创建所需目录
for d in [models_dir, logs_dir, checkpoints]:
    os.makedirs(d, exist_ok=True)

# ========== 多进程环境创建函数 ==========
def make_env(rank):
    def _init():
        try:
            gui = (rank == 0)
            env = FR5_Env(gui=gui)
            env = Monitor(env, logs_dir)
            env.reset()
            return env
        except Exception as e:
            import traceback
            print(f"[make_env-{rank}] Environment init failed:\n{traceback.format_exc()}")
            raise e
    set_random_seed(rank)
    return _init

# ========== 主函数入口 ==========
if __name__ == '__main__':
    try:
        num_train = 8
        env = SubprocVecEnv([make_env(i) for i in range(num_train)])
    except Exception:
        print("多进程创建失败，退回到 DummyVecEnv 单进程模式")
        env = DummyVecEnv([make_env(0)])

    new_logger = configure(logs_dir, ["stdout", "csv", "tensorboard"])

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=logs_dir,
        batch_size=256,
        device="cuda",
    )
    model.set_logger(new_logger)

    tensorboard_callback = TensorboardCallback()
    eval_callback = EvalCallback(
        env,
        best_model_save_path=models_dir,
        log_path=logs_dir,
        eval_freq=3000,
        deterministic=True,
        render=True,
        n_eval_episodes=100
    )

    TIMESTEPS = args.timesteps
    for episode in range(1000):
        checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=checkpoints)
        model.learn(
            total_timesteps=TIMESTEPS,
            tb_log_name=f"PPO-run-episode{episode}",
            reset_num_timesteps=False,
            callback=CallbackList([eval_callback, tensorboard_callback]),
            log_interval=10
        )
        model_path = os.path.join(models_dir, f"PPO-run-episode{episode}")
        model.save(model_path)
        logger.info(f"[Saved] PPO model saved at: {model_path}")

