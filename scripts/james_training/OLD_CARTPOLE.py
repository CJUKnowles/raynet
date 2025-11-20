import sys, os

# Add ~/raynet/build to the PATH, for access to omnetbind
sys.path.append("/home/cjuknowles/raynet/build")

#import the simulation model with cart-pole
from build.omnetbind import OmnetGymApi
import gymnasium as gym
from gymnasium import spaces, logger
import numpy as np
import math
from ray.tune.registry import register_env
import ray
from ray import tune
import random
import math
from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.rllib.algorithms.dqn.dqn import AlgorithmConfig
import time

# dumbell
# Orca

"""
This is a fully function cartpole training script with RayNet.
Do not alter - use this as a reference.
"""

class OmnetGymApiEnv(gym.Env):
    def __init__(self,env_config):
        self.action_space = spaces.Discrete(2)
        self.runner = OmnetGymApi()
        self.env_config = env_config
        self.max_episode_len = 500

        high = np.array(
            [
                2.4 * 2,
                np.finfo(np.float32).max,
                (12 * 2 * math.pi / 360) * 2,
                np.finfo(np.float32).max,],
            dtype=np.float32,)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

       
    def reset(self, *, seed=None, options=None):

        original_ini_file = self.env_config["iniPath"]

        with open(original_ini_file, 'r') as fin:
            ini_string = fin.read()
        
        ini_string = ini_string.replace("HOME",  os.getenv('HOME'))

        with open(original_ini_file + f".worker{os.getpid()}", 'w') as fout:
            fout.write(ini_string)

        self.runner.initialise(original_ini_file + f".worker{os.getpid()}")
        obs = self.runner.reset()

        obs = np.asarray(list(obs['cartpole']),dtype=np.float32)
        return  obs, {}

    def step(self, action):
        actions = {'cartpole': action}
        theta_threshold_radians = 12 * 2 * math.pi / 360
        x_threshold = 2.4
        obs, rewards, terminateds, info_ = self.runner.step(actions)
        reward = round(rewards['cartpole'],4)
        obs = obs['cartpole']

        if (obs[0] < x_threshold * -1) or (obs[0] > x_threshold) or (obs[2] < theta_threshold_radians * -1) or (obs[2] > theta_threshold_radians):
            terminateds['cartpole'] = True
            reward = 0

        if terminateds['cartpole']:
             self.runner.shutdown()
             self.runner.cleanup()
       
        obs = np.asarray(list(obs),dtype=np.float32)
    
        return  obs, reward, terminateds['cartpole'], False,{}



omnet_path = "/home/cjuknowles/raynet/build"

# Returns a training environment to use for training (defines the object that step() reset() and init() are called on)
def omnetgymapienv_creator(env_config):
    return OmnetGymApiEnv(env_config) 


if __name__ == '__main__':
    register_env("OmnetGymApiEnv", omnetgymapienv_creator)
    env = sys.argv[1]               #CartPole-v1, OmnetGymApiEnv
    num_workers = int(sys.argv[2])  # 1
    seed = int(sys.argv[3])         # 99
    random.seed(seed)
    np.random.seed(seed)

    ray.init(num_cpus=64, num_gpus=1)

    env_config = {"iniPath": os.getenv('HOME') + "/raynet/configs/cartpole/cartpole.ini"}
    #env_config={}

    # This should supposedly be replaced with AlgorithmConfig, but doesn't work
    algo = (
        DQNConfig()
        .env_runners(num_env_runners=num_workers)
        .resources(num_gpus=1)
        .environment(env, env_config=env_config) # "OmnetGymApiEnv
        .build_algo()
    )

    # Run experiments and log progress
    t_start = time.time()
    now = time.time()
    while True:
        print(f"Total elapsed: {(now - t_start)}")
        result = algo.train()
        print(result['num_env_steps_sampled_lifetime'])
        if result['num_env_steps_sampled_lifetime'] >= 10000:
            break
        now = time.time()
    ray.shutdown()
    print("Finished!")