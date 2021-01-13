import os
import random
import argparse

from multiagent.envs.multiagent_env import CustomMultiAgentEnv
from multiagent.envs.agents.basic_agent import BasicAgent
from multiagent.envs.landmarks.basic_landmark import BasicLandmark

import ray
from ray import tune
from ray.rllib.utils.test_utils import check_learning_achieved

parser = argparse.ArgumentParser()

parser.add_argument("--num-agents", type=int, default=4)
parser.add_argument("--stop-iters", type=int, default=200)
parser.add_argument("--stop-reward", type=float, default=150)
parser.add_argument("--stop-timesteps", type=int, default=100000)
parser.add_argument("--simple", action="store_true")
parser.add_argument("--num-cpus", type=int, default=0,
                    help='Number of cpus to use during training. (0 = max)')
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--framework", choices=["tf2", "tf", "tfe", "torch"], default="tf",
                    help="Framework to use during training. (default = tensorflow)")


if __name__ == "__main__":
    args = parser.parse_args()
    
    ray.init(num_cpus=args.num_cpus or None)
    
    # Register the models to use.

    a1 = BasicAgent()
    
    l1 = BasicLandmark(type="goal")
    
    # Ray config    
    config = {
        "env": CustomMultiAgentEnv,
        "env_config": {
            "agents": [a1],
            "landmarks": [l1],
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
    stop = {
        "episode_reward_mean": args.stop_reward,
        "timesteps_total": args.stop_timesteps,
        "training_iteration": args.stop_iters,
    }

    results = tune.run("PPO", stop=stop, config=config, verbose=2, checkpoint_at_end=True, checkpoint_freq=args.num_iters // 10)

    if args.as_test:
        check_learning_achieved(results, args.stop_reward)
    ray.shutdown()
