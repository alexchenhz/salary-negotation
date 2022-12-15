import argparse
import random
import ray
from ray.rllib.agents.ppo import ppo
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.tune.registry import register_env

from shutil import get_terminal_size
import pprint

import environment.job_search_environment as job_search_env
from models.job_search_model import JobSearchModelV0, TFJobSearchModelV0
from environment.job_search_environment import CANDIDATE_ACTIONS, EMPLOYER_ACTIONS
import numpy as np

tf1, tf, tfv = try_import_tf()


def get_cli_args():
    """Create CLI parser and return parsed arguments"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num-candidates", type=int, default=5, help="Number of candidate agents."
    )

    parser.add_argument(
        "--candidate-algo",
        choices=["random", "accept-first", "rl"],
        default="random",
        help="The algo for the candidate agent.",
    )

    parser.add_argument(
        "--employer-algo",
        choices=["random", "rl"],
        default="random",
        help="The algo for the employer agent.",
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
        "--checkpoint-path", type=str, help="Path to local checkpoint directory."
    )

    args = parser.parse_args()
    print(f"Running with following CLI args: {args}")
    return args


def random_action(env, obs, agent):
    agent_obs = obs[agent]
    num_actions = (
        len(CANDIDATE_ACTIONS) if "candidate" in agent else len(EMPLOYER_ACTIONS)
    )
    num_targets = max(env.num_candidates, env.num_employers)
    # Check if action mask exists
    if isinstance(agent_obs, dict) and "action_mask" in agent_obs:
        action_mask = agent_obs["action_mask"]
        legal_actions = np.flatnonzero(action_mask[0:num_actions])
        legal_targets = np.flatnonzero(
            action_mask[num_actions : num_actions + num_targets]
        )
        legal_offer_values = np.flatnonzero(
            action_mask[
                num_actions
                + num_targets : num_actions
                + num_targets
                + env.employer_budget
                + 1
            ]
        )
        legal_deadlines = np.flatnonzero(
            action_mask[
                num_actions
                + num_targets
                + env.employer_budget
                + 1 : num_actions
                + num_targets
                + env.employer_budget
                + 1
                + env.max_num_iters
                + 1
            ]
        )
        # print(legal_actions)
        action = random.choice(legal_actions) if legal_actions.any() else 0
        target = random.choice(legal_targets) if legal_targets.any() else 0
        offer_value = (
            random.choice(legal_offer_values) if legal_offer_values.any() else 0
        )
        deadline = random.choice(legal_deadlines) if legal_deadlines.any() else 0
        return action, target, offer_value, deadline
    return env.action_space(agent).sample()


if __name__ == "__main__":
    args = get_cli_args()

    ray.init()

    tf.compat.v1.enable_eager_execution()
    assert tf.executing_eagerly()

    def env_creator(config):
        env = job_search_env.env(env_config=config)
        return env

    # Register the parallel PettingZoo environment
    env_name = "job_search_env"
    register_env(env_name, lambda config: ParallelPettingZooEnv(env_creator(config)))

    # Register the models
    ModelCatalog.register_custom_model("CandidateModel", JobSearchModelV0)
    ModelCatalog.register_custom_model("EmployerModel", JobSearchModelV0)

    ModelCatalog.register_custom_model("TFCandidateModel", TFJobSearchModelV0)
    ModelCatalog.register_custom_model("TFEmployerModel", TFJobSearchModelV0)

    env_config = {
        "num_candidates": args.num_candidates,
        "num_employers": args.num_employers,
        "employer_budget": args.max_budget,
        "max_num_iters": args.max_num_iters,
    }
    env = job_search_env.env(env_config=env_config, render_mode="human")

    observations = env.reset()

    if not args.checkpoint_path:
        raise ValueError("Must specify a checkpoint path with flag --checkpoint-path")
    algo = Algorithm.from_checkpoint(
        args.checkpoint_path, ["candidate_policy", "employer_policy"]
    )

    while True:
        actions = {}
        for agent in env.agents:
            if "candidate" in agent:
                if args.candidate_algo == "random":
                    actions[agent] = random_action(env, observations, agent)
            else:
                if args.employer_algo == "random":
                    actions[agent] = random_action(env, observations, agent)
        observations, rewards, dones, _ = env.step(actions)
        print("*" * get_terminal_size()[0])
        print("Actions taken:")
        pprint.pprint(
            {
                k: (CANDIDATE_ACTIONS[v[0]],) + (v[1], v[2], v[3])
                if "candidate" in k
                else (EMPLOYER_ACTIONS[v[0]],) + (v[1], v[2], v[3])
                for k, v in actions.items()
            },
            compact=True,
        )
        env.render()

        if any([dones[key] for key in dones.keys()]):
            break

    # algo.compute_single_action(policy_id="candidate_policy")
    ray.shutdown()
