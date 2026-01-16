import sys, os
from ray.runtime_env import RuntimeEnv
from build.omnetbind import OmnetGymApi
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from ray.tune.registry import register_env
import ray
from ray import tune
import random
import math
from ray.rllib.algorithms.dqn.dqn import AlgorithmConfig
from ray.rllib.algorithms.dqn.dqn import DQNConfig
import os
import time
from random import randint

class OmnetGymApiEnv(gym.Env):
    def __init__(self,env_config):
        print("\tINIT BEING CALLED")
        self.action_space = spaces.Discrete(2)
        self.runner = OmnetGymApi()
        
        self.env_config = env_config
        self.max_episode_len = 5

        # high = np.ones(50,dtype=np.float32)* np.finfo(np.float32).max
        high = np.array(
            [
                2.4 * 2,
                np.finfo(np.float32).max,
                (12 * 2 * math.pi / 360) * 2,
                np.finfo(np.float32).max,],
            dtype=np.float32,)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
       
    def reset(self, *, seed=None, options=None):
        print("\tRESET BEING CALLED")

        original_ini_file = self.env_config["iniPath"]
        print("\tRESET BEING CALLED 1")
        # Replace HOME with absolute paths in the simulation ini file
        with open(original_ini_file, 'r') as fin:
            ini_string = fin.read().replace("HOME",  os.getenv('HOME'))
        with open(original_ini_file + f".worker{os.getpid()}", 'w') as fout:
            fout.write(ini_string)
        print("\tRESET BEING CALLED 2" )
        # Start a new simulation runner on the modified ini file
        self.runner.initialise(original_ini_file + f".worker{os.getpid()}")
        print("\tRESET BEING CALLED 3")
        obs = self.runner.reset()
        print("\tRESET BEING CALLED 4")
        print(obs)
        obs = np.asarray(list(obs['JamesCC']),dtype=np.float32)
        print(obs)
        return  obs, {}

    def step(self, action):
        print("\tSTEP BEING CALLED")
        actions = {'JamesCC': action}

        obs, rewards, terminateds, info_ = self.runner.step(actions)
        if terminateds['JamesCC']:
             self.runner.shutdown()
             self.runner.cleanup()
             
        obs = obs['JamesCC']
        obs = np.asarray(list(obs),dtype=np.float32)
        reward = randint(0,1) # Reward value, random for now to see if the value actually reaches the agent
        # OBS, REWARD, IS_TERMINATED, IS_TRUNCATED, EXTRA_INFO
        return  obs, reward, terminateds['JamesCC'], False,{"test": "this is a test! Can the JamesCC see this info?"}


# Generates the OmnetGymApiEnv for the calling ray worker
def omnetgymapienv_creator(env_config):
    return OmnetGymApiEnv(env_config)  # return an env instance

register_env("OmnetGymApiEnv", omnetgymapienv_creator)

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        #raise Exception("This script expects arguments ENV, NUM_WORKERS, SEED. Please provide arguments or use a runner.py")
        env = "OmnetGymApiEnv"
        num_workers = 1
        seed = 918284
    else:
        env = sys.argv[1]               # OmnetGymApiEnv, CartPole-v1
        num_workers = int(sys.argv[2])  # 1, 2, 4, 8, 16
        seed = int(sys.argv[3])         # any num
    
    random.seed(seed)
    np.random.seed(seed)

    ray.init(num_cpus=64)

    env_config = {"iniPath": os.getenv('HOME') + "/raynet/configs/james/james.ini"}

    algo = (
            DQNConfig()
            .resources(num_gpus=0)
            .env_runners(num_env_runners=num_workers, num_gpus_per_env_runner=0)
            .environment(env, env_config=env_config) # "OmnetGymApiEnv
            .build_algo()
            )
    
    # Run experiments and log progress
    t_start = time.time()
    now = time.time()
    while True:
        print(f"Total elapsed: {(now - t_start)}")
        print("before")
        result = algo.train()
        print("after")
        # for i in range(0, 20):
        #     print("Result object:")
        # for i in result:
        #     print(i, result[i])
        #     print()
        print(result['num_env_steps_sampled_lifetime'])
        if result['num_env_steps_sampled_lifetime'] >= 2000:
            break
        now = time.time()
    ray.shutdown()
    print("Finished!")