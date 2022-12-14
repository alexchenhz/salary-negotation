import argparse
import os

import ray
from gym.spaces import Box, Discrete
from ray import air, tune
from ray.rllib.algorithms import ppo
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import ppo

import environment.job_search_environment as job_search_env
from models.job_search_model import JobSearchModelV0


def get_cli_args():
    """Create CLI parser and return parsed arguments"""
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--run", type=str, default="PPO", help="The RLlib-registered algorithm to use."
    )
    parser.add_argument("--num-cpus", type=int, default=0)
    
    parser.add_argument("--num-gpus", type=int, default=0)

    parser.add_argument("--eager-tracing", action="store_true")
    
    parser.add_argument(
        "--stop-iters", type=int, default=5, help="Number of iterations to train."
    )
    parser.add_argument(
        "--stop-timesteps",
        type=int,
        default=10000,
        help="Number of timesteps to train.",
    )
    parser.add_argument(
        "--stop-reward",
        type=float,
        default=1000,
        help="Reward at which we stop training.",
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Run without Tune using a manual train loop instead. Here,"
        "there is no TensorBoard support.",
    )
    parser.add_argument(
        "--local-mode",
        action="store_true",
        help="Init Ray in local mode for easier debugging.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples to run concurrently"
    )

    args = parser.parse_args()
    print(f"Running with following CLI args: {args}")
    return args


if __name__ == "__main__":
    args = get_cli_args()

    ray.init(num_cpus=args.num_cpus or None, local_mode=args.local_mode)

    def env_creator(args):
        env = job_search_env.env()
        return env

    # Register the parallel PettingZoo environment
    env_name = "job_search_env"
    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))

    # Register the model
    ModelCatalog.register_custom_model("JobSearchModelV0", JobSearchModelV0)

    config = (
        PPOConfig()
        .environment(env=env_name, clip_actions=False)
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        .resources(num_gpus=args.num_gpus)
        .training(
            model={
                "custom_model": JobSearchModelV0,
                "custom_model_config": {},
            }
        )
    ).to_dict()

    # Set observation space and action space (all agents should have the same spaces)
    config["observation_space"] = job_search_env.env().observation_space("candidate_0")
    config["action_space"] = job_search_env.env().action_space("candidate_0")

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        # "episode_reward_mean": args.stop_reward,
    }

    tune.run(
        ppo.PPOTrainer,
        stop=stop,
        checkpoint_freq=5,
        local_dir="./ray_results/" + env_name,
        config=config,
        num_samples=args.num_samples,
    )
    ray.shutdown()
