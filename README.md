# Salary negotiation

## Job search environment

`possible_agents`

Agents should be the list of strings. With either candidate agents or employer agents. To start, let's say there are 3 candidate agents and 3 employer agents.

`observations`

Each candidate agent should be able to observe

- job openings (which employer agents are still hiring)
- their current offers (with offer value and expiration)
- their rejected offers (with offer value)
- their counter offers (counter offer value, also store original offer details)

Each employer agent should be able to observe

- job applicants (which candidates have applied for the job)
- outstanding offers (candidate, offer value, and expiration)
- declined offers (candidate, offer value)
- counter offers (new offers made from candidates, with offer value)
- rejected counter offers (counter offers the employer agent has rejected)
- remaining budget (employer will have a limitted amount of resoures to allocate across all job offers, cannot pay everyone as high of a number as possible)

`step`

At each step, agents should take an action.

Candidate actions:

- Apply to job
- Accept offer
- Negotiate offer (make a counter-offer)
- Reject offer

Possible actions for employers:

- Reject applicant
- Make offer (or make counter counter-offer)
- Accept counter-offer
- Reject counter-offer

How many actions can employers/candidates do at once?

- There should be some kind of penalty, since these actions take time.
- Also, some actions should only be available if certain observed conditions are met.
- 1 action per step -> penalty will be others could accept an offer before you do
- Each time step you aren't working you get a pentaly
- Each time step an employer doesn't have an employee it gets a penalty
- Offers with deadlines (2 full iterations of all agents?)

Rewards

1. At each step, the agent can choose an action

## Agents

Start with stable baselines RL algorithms (ex: PPO).
