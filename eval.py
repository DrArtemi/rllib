import os
import argparse

from multiagent.envs.multiagent_env import CustomMultiAgentEnv
from multiagent.envs.agents.basic_agent import BasicAgent
from multiagent.envs.landmarks.basic_landmark import BasicLandmark

import ray
import ray.rllib.agents.ppo as ppo

parser = argparse.ArgumentParser()

parser.add_argument("--num-steps", type=int, default=300)
parser.add_argument("--simple", action="store_true")
parser.add_argument("--num-cpus", type=int, default=0,
                    help='Number of cpus to use during training. (0 = max)')
parser.add_argument("--framework", choices=["tf2", "tf", "tfe", "torch"], default="tf",
                    help="Framework to use during training. (default = tensorflow)")
parser.add_argument("--checkpoint", type=str, default="")


if __name__ == "__main__":
    args = parser.parse_args()
    
    ray.init(num_cpus=args.num_cpus or None)
    
    a1 = BasicAgent(
        initial_conditions={
            "position": [0.8, 0.2],
            "color": [0, 0, 1, 1]
        })
    
    l1 = BasicLandmark(
        type="goal",
        initial_conditions={
            "color": [0, 1, 0, 1]
        })
    
    # Ray config
    config = ppo.DEFAULT_CONFIG.copy()
    config = {
        "env": CustomMultiAgentEnv,
        "env_config": {
            "agents": [a1],
            "landmarks": [l1],
            "max_episode_step": 100
        },
        "simple_optimizer": args.simple,
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "num_sgd_iter": 10,
        "multiagent": {
        "policies": {
            "basic_agent": (None, a1.observation_space, a1.action_space, {"gamma": 0.95}),
        },
        "policy_mapping_fn":
            lambda agent_id: "basic_agent"
        },
        "framework": args.framework,
    }
    trainer = ppo.PPOTrainer(config=config)
    trainer.restore(args.checkpoint)
    
    env = CustomMultiAgentEnv(config={
        "agents": [a1],
        "landmarks": [l1],
        "max_episode_step": 100
    })
    state = env.reset()
    
    for step in range(1, args.num_steps + 1):
        actions = dict()
        for agent in env.agents:
            actions[agent.id] = trainer.compute_action(state[agent.id], policy_id="basic_agent")
        state, reward, done, info = env.step(actions)
        env.render()
        
        if done['__all__']:
            state = env.reset()
    
    ray.shutdown()
