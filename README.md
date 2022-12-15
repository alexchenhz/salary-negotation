# Applying Multi Agent Reinforcement Learning to Candidate/Employer Job Matching and Salary Negotiations

Alexander H. Chen

Yale University

CSEC 491 Senior Project

December 15, 2022

Thank you to Dr. James Glenn for advising me on this project.

## Abstract

In this project, we explore the use of reinforcement learning to train candidate and employer agents to choose actions that maximize their respective payoffs in the job search and salary negotiation process. To do this, we first used the PettingZoo open source library to create a multi agent reinforcement learning environment that models this process. Breaking down the job search and salary negotiation process into steps, each candidate agent can choose to apply to a position, accept an offer, reject an offer, or negotiate an offer, and each employer agent can choose to reject an applicant, make an offer, accept a counter offer, or reject a counter offer. Each agent also has its own observations, which reflect an agent’s knowledge of the overall game state. This environment allowed us to simulate the interactions between candidate and employer agents as they make decisions and negotiate salaries based on their objectives and rewards. Next, we used the Ray RLlib open source library to train reinforcement learning agents to optimize their decision-making in this environment. The candidate agents were trained to maximize their offer values, while the employer agents were trained to maximize the difference between candidate strengths and offer values. Our results show that these trained agents exhibit improved decision-making compared to random agents and simple strategy agents, resulting in an increase in reward value. This suggests that reinforcement learning can be a powerful tool for modeling and optimizing the job search and salary negotiation process. This project lays the groundwork for further experimentation and modeling of the job matching process.

## Final Project Report PDF



## Original Project Description PDF

[CSEC 491 Project Description - Alexander Chen](./CSEC%20491%20Project%20Description%20-%20Alexander%20Chen.pdf)

## Repository README

### Getting started

Activate the virtual environment.

```bash
source venv/bin/activate

```

Install the required packages. This project primarily depends on `pettingzoo` and `ray`, in addition to other packages.

```bash
pip install -r requirements.txt
```

Train the agents. See the `job_search.py` file for all CLI flags and args. Note this assumes the training is run on the Zoo with 16 CPU cores (using 4 workers + 1 local worker per trial, so requires 5 CPU cores per trial running in parallel). I also had issues running Ray on my M1 MacBook, so best to stick to x86 for now.

```bash
python job_search.py --num-candidates <int> --num-employers <int> --max-budget <int> --max-num-iters <int>
```

This will create a new directory `ray_results` which will store the information from training the reinforcement learning policies.

By default, this will use the TensorFlow job search model. Use TensorBoard to view the training metrics.

```bash
tensorboard --logdir ray_results/job_search_env/<path to results>
```

### Simulate a job search game play-through

Get the path to the latest checkpoint file of the training you want to use.

```bash
python job_search_simulation.py --checkpoint-path ray_results/job_search_env/.../checkpoint_XXXXXX --num-candidates <int> --num-employers <int> --max-budget <int> --max-num-iters <int> --candidate-algo <random/rl> --employer-algo <random/rl>
```

Note, you will want to ensure the parameters for the environment are the same as the ones used to train the model your are using.

### Repository structure

```bash
    .
    ├── archive
    ├── environment
    │   ├── job_search_environment.py
    ├── index.md
    ├── job_search.py
    ├── job_search_simulation.py
    ├── models
    │   ├── job_search_model.py
    ├── ray_results
    ├── README.md
    ├── requirements.txt
    └── venv
```

#### `job_search_environment.py`

`game_state` (observations)

Each candidate agent should be able to observe:

- job openings (which employer agents are still hiring)
- their current offers (with offer value and expiration)
- their rejected offers (with offer value)
- their counter offers (counter offer value, also store original offer details)

Each employer agent should be able to observe:

- candidate strengths (after candidate applies, store how strong the candidate is)
- job applicants (which candidates have applied for the job)
- outstanding offers (candidate, offer value, and expiration)
- declined offers (candidate, offer value)
- counter offers (new offers made from candidates, with offer value)
- rejected counter offers (counter offers the employer agent has rejected)
- remaining budget (employer will have a limitted amount of resoures to allocate across all job offers, cannot pay everyone as high of a number as possible)

`step`

At each step, agents should take an action.

Candidate actions:

- No action
- Apply to job
- Accept offer
- Decline offer
- Negotiate offer (make a counter-offer)

Employer actions:

- No action
- Reject applicant
- Make offer (or make counter counter-offer)
- Accept counter-offer
- Reject counter-offer

Each agent can only execute one action per `step()`.

Expired offers or counter offers will be removed automatically each `step()` and considered rejected/declined.

##### Rewards

A candidate agent will receive a reward equal to the value of their accepted offer divided by the discount rate raised to the power of the number of iterations of the game that have passed.

$$r_{c} = v_{o} / (1 + r)^{t}$$

An employer agent will receive a reward equal to the strength of the candidate minus the value of the offer, all divided by the discount rate raised to the power of the number of iterations of the game that have passed.

$$r_{e} = (s_{c} - v_{o})/(1 + r)^{t}$$

We will assume the discount rate to be 0.05.
