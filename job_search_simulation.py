import argparse

import ray
from ray.rllib.agents.ppo import ppo
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.tune.registry import register_env

import environment.job_search_environment as job_search_env
from models.job_search_model import JobSearchModelV0, TFJobSearchModelV0

tf1, tf, tfv = try_import_tf()


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
        "--checkpoint-path", type=str, help="Path to local checkpoint directory."
    )

    args = parser.parse_args()
    print(f"Running with following CLI args: {args}")
    return args


class JobSearchSimulation:
    def load(self, path):
        """
        Load a trained RLlib agent from the specified path. Call this before testing a trained agent.
        :param path: Path pointing to the agent's saved checkpoint (only used for RLlib agents)
        """
        self.agent = ppo.PPOTrainer(config=self.config, env=self.env_class)
        self.agent.restore(path)

    def test(self):
        """Test trained agent for a single episode. Return the episode reward"""
        # instantiate env class
        env = self.env_class(self.env_config)

        # run until episode ends
        episode_reward = 0
        done = False
        obs = env.reset()
        while not done:
            action = self.agent.compute_action(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward

        return episode_reward


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

    env.reset()

    if not args.checkpoint_path:
        raise ValueError("Must specify a checkpoint path with flag --checkpoint-path")
    algo = Algorithm.from_checkpoint(
        args.checkpoint_path, ["candidate_policy", "employer_policy"]
    )
    # algo.compute_single_action(policy_id="candidate_policy")
    ray.shutdown()
