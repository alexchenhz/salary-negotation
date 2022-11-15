# Salary negotiation

## Job search environment

`possible_agents`

Agents should be the list of strings. With either candidate agents or employer agents. To start, let's say there are 3 candidate agents and 3 employer agents.

`observations`

Each candidate agent should be able to observe

- the number of employer agents still hiring
- their current offers and their values
- their previous offers and their values

Each employer agent should be able to observe

- the number of candidates still applying
- how many offers remaining
- how much budget remaining
- who rejected an offer + details of that rejected offer

`step`

At each step, agents should take an action.

Possible actions for candidates:

- Apply to job
- Accept offer
- Negotiate offer (counter-offer)
- Reject offer

Possible actions for employers:

- Reject candidate
- Make offer
- Accept counter-offer
- Make counter counter-offer
- Reject counter-offer

How many actions can employers/candidates do at once?

- There should be some kind of penalty, since these actions take time.
- Also, some actions should only be available if certain observed conditions are met.
- 1 action per step -> penalty will be others could accept an offer before you do
- Each time step you aren't working you get a pentaly
- Each time step an employer doesn't have an employee it gets a penalty
- Offers with deadlines (2 full iterations of all agents?)

# Agents

Start with stable baselines RL algorithms (ex: PPO).
