"""
A test for using TD3 to train Orca, without Ray/RLlib.
Currently does not work. Existing models were trained with SAC on Ray/RLlib.
"""
import argparse
import math
import os
import random
import sys
from collections import deque

import gymnasium as gym
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise


class OmnetGymApiEnv(gym.Env):
    def __init__(self, env_config):
        sys.path.insert(0, os.path.join(os.getenv("HOME"), "raynet", "build"))
        from omnetbind import OmnetGymApi

        self.runner = OmnetGymApi()
        self.env_config = env_config
        self.step_count = 0
        self.random_seed = os.getpid()
        random.seed(self.random_seed)

        self.bw = self.env_config["bottleneck_bw_range"][0]
        self.base_rtt = self.env_config["minimum_rtt_range"][1]
        self.buffer_size = self.env_config["bottleneck_buffer_range"][0]
        self.max_steps_range = self.env_config["max_steps_range"][1]
        self.stacking = self.env_config["stacking"]
        self.num_observations = 7

        self.action_space = gym.spaces.Box(low=-2, high=2, shape=(1,), dtype=np.float32)

        self.obs_min = np.tile(
            np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32), self.stacking
        )
        self.obs_max = np.tile(
            np.array([1, 10, 10, 10, 1, 1, 1], dtype=np.float32), self.stacking
        )
        self.observation_space = gym.spaces.Box(
            low=self.obs_min,
            high=self.obs_max,
            dtype=np.float32,
        )
        self.obs_history = deque(
            np.zeros(self.stacking * self.num_observations, dtype=np.float32),
            maxlen=self.stacking * self.num_observations,
        )

    def reset(self, *, seed=None, options=None):
        
        del seed, options
        self.obs_history = deque(
            np.zeros(self.stacking * self.num_observations, dtype=np.float32),
            maxlen=self.stacking * self.num_observations,
        )

        bottleneck_bw_range = self.env_config["bottleneck_bw_range"]
        base_rtt_range = self.env_config["minimum_rtt_range"]
        bottleneck_buffer_range = self.env_config["bottleneck_buffer_range"]
        max_steps_range = self.env_config["max_steps_range"]

        self.bw = round(np.random.uniform(low=bottleneck_bw_range[0], high=bottleneck_bw_range[1]))
        self.base_rtt = round(np.random.uniform(low=base_rtt_range[0], high=base_rtt_range[1]), 2)
        self.buffer_size = round(
            np.random.uniform(low=bottleneck_buffer_range[0], high=bottleneck_buffer_range[1])
        )
        self.max_steps = round(np.random.uniform(low=max_steps_range[0], high=max_steps_range[1]))

        original_ini_file = self.env_config["iniPath"]
        ini_dir, ini_file = original_ini_file.rsplit("/", 1)
        ini_variants_dir = f"{ini_dir}/ini_variants"
        os.makedirs(ini_variants_dir, exist_ok=True)
        worker_ini_file = f"{ini_variants_dir}/{ini_file}.worker{os.getpid()}"

        with open(original_ini_file, "r", encoding="utf-8") as fin:
            ini_string = fin.read()
        ini_string = ini_string.replace("HOME", os.getenv("HOME"))
        ini_string = ini_string.replace("ORCA_BOTTLENECK_BW", f"{self.bw}Mbps")
        ini_string = ini_string.replace("ORCA_BASE_RTT", f"{self.base_rtt / 2.0}ms")
        ini_string = ini_string.replace("ORCA_BOTTLENECK_BUFFER_SIZE", f"{self.buffer_size}b")
        ini_string = ini_string.replace("MAX_RL_STEPS", f"{self.max_steps}")

        print(f"SAVING TO {worker_ini_file}")
        with open(worker_ini_file, "w", encoding="utf-8") as fout:
            fout.write(ini_string)
        print("RESETINFO: INIT BEING CALLED")
        self.runner.initialise(worker_ini_file, "General")
        print("RESETINFO: INIT COMPLETE")
        print("RESETINFO: RESET BEING CALLED")
        obs = self.runner.reset()["Orca"]
        print("RESETINFO: RESET COMPLETE")
        for _ in range(self.stacking):
            self.obs_history.extend(obs)
        self.obs_history.extend(obs)
        return np.asarray(list(self.obs_history), dtype=np.float32), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action = {"Orca": float(action.item())}
        obs, rewards, terminateds, info_ = self.runner.step(action)
        self.obs_history.extend(obs["Orca"])

        obs_history = np.asarray(list(self.obs_history), dtype=np.float32)
        reward = float(rewards["Orca"])
        terminated = bool(terminateds["Orca"])
        truncated = bool(info_.get("simDone", False))

        if math.isnan(reward):
            print("Warning: NaN reward returned, replacing with 0.0")
            reward = 0.0

        if terminated:
            print("TERMINATED")
            self.runner.shutdown()
            self.runner.cleanup()
        if truncated:
            print("TRUNCATED")
        return obs_history, reward, terminated, truncated, {}

    def close(self):
        try:
            self.runner.shutdown()
            self.runner.cleanup()
        except Exception:
            pass


def parse_args():
    parser = argparse.ArgumentParser(description="Orca TD3 training with Stable-Baselines3")
    parser.add_argument("--seed", type=int, default=91456211)
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--policy-noise", type=float, default=0.2)
    parser.add_argument("--noise-clip", type=float, default=0.5)
    parser.add_argument("--exploration-noise", type=float, default=0.1)
    parser.add_argument("--learning-starts", type=int, default=25_000)
    parser.add_argument("--policy-delay", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=50_000)
    parser.add_argument("--log-dir", type=str, default="")
    parser.add_argument("--checkpoint-path", type=str, default="")
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    env_config = {
        "iniPath": os.getenv("HOME") + "/raynet/simlibs/Orca/src/training/OrcaTraining.ini",
        "bottleneck_bw_range": (5, 20),
        "minimum_rtt_range": (5, 100),
        "bottleneck_buffer_range": (25000, 4_000_000),
        "max_steps_range": (2000, 2000),
        "stacking": 10,
    }

    random.seed(args.seed)
    np.random.seed(args.seed)

    run_name = f"orca_td3_{args.seed}"
    log_dir = args.log_dir or os.path.join("runs", run_name)
    os.makedirs(log_dir, exist_ok=True)
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    env = Monitor(OmnetGymApiEnv(env_config))
    action_dim = int(np.prod(env.action_space.shape))
    action_noise = NormalActionNoise(
        mean=np.zeros(action_dim, dtype=np.float32),
        sigma=args.exploration_noise * np.ones(action_dim, dtype=np.float32),
    )

    policy_kwargs = dict(net_arch=dict(pi=[256, 256], qf=[256, 256]))
    device = "cuda" if args.cuda else "auto"

    if args.checkpoint_path:
        model = TD3.load(
            args.checkpoint_path,
            env=env,
            device=device,
        )
        model.action_noise = action_noise
    else:
        model = TD3(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            gamma=args.gamma,
            tau=args.tau,
            #policy_noise=args.policy_noise,
            #noise_clip=args.noise_clip,
            learning_starts=args.learning_starts,
            policy_delay=args.policy_delay,
            action_noise=action_noise,
            train_freq=(1, "step"),
            gradient_steps=1,
            policy_kwargs=policy_kwargs,
            tensorboard_log=log_dir,
            verbose=1,
            device=device,
        )

    checkpoint_callback = None
    if args.save_every and args.save_every > 0:
        checkpoint_callback = CheckpointCallback(
            save_freq=args.save_every,
            save_path=checkpoint_dir,
            name_prefix="orca_td3",
            save_replay_buffer=True,
            save_vecnormalize=True,
        )

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=checkpoint_callback,
            progress_bar=False,
        )
    finally:
        final_model_path = os.path.join(log_dir, "orca_td3_final")
        model.save(final_model_path)
        env.close()
        print(f"Saved final model to {final_model_path}.zip")


if __name__ == "__main__":
    main()
