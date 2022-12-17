import argparse
import os

import ray
from ray import tune
from ray.rllib.agents.ppo import ppo
from ray.rllib.algorithms.ppo import PPOConfig, PPOTorchPolicy, PPOTF2Policy
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

import environment.job_search_environment as job_search_env
from models.job_search_model import JobSearchModelV0, TFJobSearchModelV0


def get_cli_args():
    """Create CLI parser and return parsed arguments"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num-candidates", type=int, default=5, help="Number of candidate agents."
    )

    parser.add_argument(
        "--num-employers", type=int, default=5, help="Number of employer agents."
    )

    parser.add_argument(
        "--max-num-iters",
        type=int,
        default=10,
        help="Maximum number of iterations for the job search environment.",
    )

    parser.add_argument(
        "--max-budget", type=int, default=100, help="Maximum budget for each employer."
    )

    parser.add_argument(
        "--run", type=str, default="PPO", help="The RLlib-registered algorithm to use."
    )
    parser.add_argument("--num-cpus", type=int, default=0)
    parser.add_argument(
        "--framework",
        choices=["tf2", "torch"],
        default="tf2",
        help="The DL framework specifier.",
    )
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
        "--checkpoint-freq",
        type=int,
        default=1,
        help="Number of iterations between checkpoint saves",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples to run concurrently",
    )

    args = parser.parse_args()
    print(f"Running with following CLI args: {args}")
    return args


if __name__ == "__main__":
    args = get_cli_args()

    ray.init(num_cpus=args.num_cpus or None)

    def env_creator(config):
        env = job_search_env.env(env_config=config)
        return env

    # Register the parallel PettingZoo environment
    env_name = "job_search_env"
    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))

    env_config = {
        "num_candidates": args.num_candidates,
        "num_employers": args.num_employers,
        "employer_budget": args.max_budget,
        "max_num_iters": args.max_num_iters,
    }

    # Get observation space and action space from environment (all agents should have the same spaces)
    obs_space = job_search_env.env(env_config=env_config).observation_space(
        "candidate_0"
    )
    act_space = job_search_env.env(env_config=env_config).action_space("candidate_0")

    # Register the models
    ModelCatalog.register_custom_model("CandidateModel", JobSearchModelV0)
    ModelCatalog.register_custom_model("EmployerModel", JobSearchModelV0)

    ModelCatalog.register_custom_model("TFCandidateModel", TFJobSearchModelV0)
    ModelCatalog.register_custom_model("TFEmployerModel", TFJobSearchModelV0)

    if args.framework == "torch":
        policies = {
            "candidate_policy": (
                PPOTorchPolicy,
                obs_space,
                act_space,
                {
                    "model": {
                        "custom_model": "CandidateModel",
                        "custom_model_config": {},
                    }
                },
            ),
            "employer_policy": (
                PPOTorchPolicy,
                obs_space,
                act_space,
                {
                    "model": {
                        "custom_model": "EmployerModel",
                        "custom_model_config": {},
                    }
                },
            ),
        }
    else:
        policies = {
            "candidate_policy": (
                PPOTF2Policy,
                obs_space,
                act_space,
                {
                    "model": {
                        "custom_model": "TFCandidateModel",
                        "custom_model_config": {},
                    }
                },
            ),
            "employer_policy": (
                PPOTF2Policy,
                obs_space,
                act_space,
                {
                    "model": {
                        "custom_model": "TFEmployerModel",
                        "custom_model_config": {},
                    }
                },
            ),
        }

    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        return "candidate_policy" if "candidate" in agent_id else "employer_policy"

    config = (
        PPOConfig()
        .environment(
            env=env_name,
            clip_actions=False,
            env_config=env_config,
        )
        .rollouts(num_rollout_workers=4)
        .debugging(log_level="ERROR")
        .framework(framework=args.framework)
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
        .training(
            model={
                # "custom_model": JobSearchModelV0 if args.framework == "torch" else TFJobSearchModelV0,
                # "custom_model_config": {},
            }
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["candidate_policy", "employer_policy"],
        )
    ).to_dict()

    # Set observation space and action space (all agents should have the same spaces)
    config["observation_space"] = obs_space
    config["action_space"] = act_space

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
    }

    env_info = f"c{args.num_candidates}e{args.num_employers}b{args.max_budget}i{args.max_num_iters}"

    tune.run(
        ppo.PPOTrainer,
        stop=stop,
        checkpoint_freq=args.checkpoint_freq,
        local_dir="./ray_results/{}/{}/{}".format(env_name, args.framework, env_info),
        config=config,
        num_samples=args.num_samples,
    )
    ray.shutdown()
