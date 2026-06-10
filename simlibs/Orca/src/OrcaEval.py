import sys
import os
import random
from collections import deque, defaultdict
import numpy as np
import ray
import torch
from gymnasium import spaces
from ray.tune.registry import register_env
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.algorithms.sac.sac import SACConfig
from ray.rllib.core.columns import Columns

class OmnetGymApiEnv(MultiAgentEnv):
    def __init__(self, env_config):
        super().__init__()
        sys.path.insert(0, os.path.join(os.getenv("RAYNET_PATH"), "build"))
        from omnetbind import OmnetGymApi
        self.runner = OmnetGymApi()
        self.env_config = env_config
        self.stacking = env_config["stacking"]
        self.obs_size = 7
        self.random_seed = os.getpid()
        random.seed(self.random_seed)
        # Observation bounds
        single_obs_min = np.array(
            [
                0,  # throughput
                0,  # pacerate
                0,  # lossrate
                0,  # ack count
                0,  # interval duration
                0,  # srtt
                0,  # delay metric
            ],
            dtype=np.float32,
        )

        single_obs_max = np.array(
            [
                1,
                10,
                10,
                10,
                1,
                1,
                1,
            ],
            dtype=np.float32,
        )

        self.obs_min = np.tile(single_obs_min, self.stacking)
        self.obs_max = np.tile(single_obs_max, self.stacking)

        # IMPORTANT:
        # MultiAgentEnv spaces describe ONE agent.
        self.observation_space = spaces.Box(
            low=self.obs_min,
            high=self.obs_max,
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-2,
            high=2,
            shape=(1,),
            dtype=np.float32,
        )

        # Maximum possible agents
        self.possible_agents = [
            f"Orca{i}"
            for i in range(env_config["num_flows_range"][1])
        ]

        self.agents = []

    def reset(self, *, seed=None, options=None):
        self.agents = []
        self.obs_history = defaultdict(
            lambda: deque(
                [np.zeros(self.obs_size, dtype=np.float32) for _ in range(self.stacking)],
                maxlen=self.stacking,
            )
        )

        self.runner.initialise(self.env_config["iniPath"], self.env_config["config_section"],)
        raw_obs = self.runner.reset()
        obs = {}

        for agent_id, agent_obs in raw_obs.items():
            if agent_id == "__all__":
                continue
            self.agents.append(agent_id)
            self.obs_history[agent_id].append(np.asarray(agent_obs, dtype=np.float32))
            stacked_obs = np.concatenate(self.obs_history[agent_id]).astype(np.float32)
            obs[agent_id] = stacked_obs

        infos = {agent_id: {} for agent_id in obs}
        print(f"Initial agents: {self.agents}")
        return obs, infos

    def step(self, actions):
        converted_actions = {}
        for agent_id, action in actions.items():
            converted_actions[agent_id] = float(np.clip(np.asarray(action).item(), -2, 2))
            
        raw_obs, rewards, terminateds, info_ = self.runner.step(converted_actions)
        
        print(f"raw_obs:\n{raw_obs}")
        
        obs = {}
        for agent_id, agent_obs in raw_obs.items():
            if agent_id == "__all__":
                continue
            self.obs_history[agent_id].append(np.asarray(agent_obs, dtype=np.float32))
            stacked_obs = np.concatenate(self.obs_history[agent_id]).astype(np.float32)
            obs[agent_id] = stacked_obs

        if terminateds["__all__"]:
            print("Episode complete (terminated).")
            self.runner.shutdown()
            self.runner.cleanup()
        elif info_["simDone"]:
            print("Episode complete (truncated).")
            self.runner.cleanup()
            truncateds = {agent_id: True for agent_id in obs}
            truncateds["__all__"] = True
        else:
            truncateds = {agent_id: False for agent_id in obs}
            truncateds["__all__"] = False

        

        infos = {agent_id: {} for agent_id in obs}
        print(f"Step obs histories:\n {self.obs_history}")
        return (obs, rewards, terminateds, truncateds, infos,)

def omnetgymapienv_creator(env_config):
    return OmnetGymApiEnv(env_config)

if __name__ == "__main__":
    env_name = "Orca-Eval"
    register_env(env_name, omnetgymapienv_creator)
    stacking = 10

    env_config = {
        "iniPath": sys.argv[1],
        "config_section": (
            sys.argv[2]
            if len(sys.argv) > 2
            else "Orca"
        ),
        "stacking": stacking,
        "bottleneck_bw_range": (5, 20),
        "minimum_rtt_range": (5, 100),
        "bottleneck_buffer_range": (25000, 2000000),
        "max_steps_range": (2000, 2000),
        "num_flows_range": (2, 5),
    }

    checkpoint_load_dir = (
        os.getenv("RAYNET_PATH")
        + "/_models/Orca-aiden-10k"
    )

    ray.init(
        local_mode=True,
        include_dashboard=False,
        ignore_reinit_error=True,
        _temp_dir=f"/tmp/ray_{os.getpid()}",
        num_cpus=1,
    )

    config = (
        SACConfig()
        .environment(
            env=env_name,
            env_config=env_config,
        )
        .multi_agent(
            policies={"default_policy"},
            policy_mapping_fn=lambda agent_id, *args, **kwargs:
                "default_policy",
            policies_to_train=[],
        )
    )

    algo = config.build()
    algo.restore(checkpoint_load_dir)
    print("Checkpoint restored.")

    # Get RLModule directly (new API stack)
    module = algo.get_module("default_policy")
    env = OmnetGymApiEnv(env_config)
    obs, infos = env.reset()
    terminateds = {"__all__": False}
    total_rewards = defaultdict(float)

    while not terminateds["__all__"]:
        actions = {}
        for agent_id, agent_obs in obs.items():
            batch = {
                Columns.OBS: torch.from_numpy(
                    np.expand_dims(agent_obs, axis=0)
                ).float()
            }
            with torch.no_grad():
                output = module.forward_inference(batch)
                action_dist_class = (module.get_inference_action_dist_cls())
                action_dist = action_dist_class.from_logits(output["action_dist_inputs"])
                action = (action_dist.to_deterministic().sample()[0])

            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()

            actions[agent_id] = action

        (obs, rewards, terminateds, truncateds, infos,) = env.step(actions)
        for agent_id, reward in rewards.items():
            total_rewards[agent_id] += reward

        print(f"Step rewards: {rewards}")
    print("\nEvaluation complete.")

    for agent_id, reward in total_rewards.items():
        print(f"{agent_id}: {reward}")