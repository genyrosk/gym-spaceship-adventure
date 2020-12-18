# Spacehip Adventure environment for OpenAI gym 🚀

A cool variation on the [frozen-lake](https://gym.openai.com/envs/FrozenLake-v0/) environment.

The agent controls the movement of a spaceship in a grid world.
Some tiles of the grid are navigable, and others lead to the agent colliding with an asteroid 💥.
Additionally, the agent can activate a boost ⚡ and move accross 4 tiles in one go.
The agent is rewarded for finding a navigable path to a goal tile.

The space is described using a grid like the following:

```

  AA-A-G
  --A--x         (G: goal tile)
  -x--AA         (-: empty space, safe)
  --A--A         (x : pseudo-reward tile, safe)
  A--A--         (A: asteroid, collide to your doom)
  S-----         (S: starting point, safe)

```

The episode ends when you reach the goal or collide with an asteroid.
By default, you receive a reward of -1 for each movement, 1 for reaching a pseudo-reward, 10 if you reach the goal, and -10 if you collide.

To improve training and exploration you can configure a pseudo-reward policy (c.f. [Policy invariance under reward transformations](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf) by Ng, Harada and Russell).


# Installation

```bash
git clone https://github.com/genyrosk/gym-spaceship-adventure
cd gym-spaceship-adventure
pip install -e .
```

# Usage

Note: Generated maps are guaranteed to have a solution from the starting point to the goal tile.

```python

import gym

# Default map
env = gym.make("spaceship-adventure-v0")

# Randomly generated map of a certain size and probability of empty tile `p`
env = gym.make("spaceship-adventure-v0", map_name=None, size=6, p=0.7)

# get available actions
possible_actions = env.get_possible_actions()

# take a step
state, reward, done, info = env.step(3)

# render the current state
env.render()

# reset
env.reset()

```


# Parametrization

Use a pre-defined map:

```python

env = gym.make(
  "spaceship-adventure-v0",
  map_name="6x6",
)

```

Generate a new map every time:

```python

env = gym.make(
  "spaceship-adventure-v0",
  map_name=None, # important if you want to generate a new map
)

```

The simulation can be highly customized.

```python

env = gym.make(
  "spaceship-adventure-v0",
  map_name=None, # important if you want to generate a new map
  size=10,
  p=0.7,
  max_boosts=5,
  action_randomness=0.2,
  movement_reward=-1,
  asteroid_reward=-10,
  goal_reward=10,
  num_pseudo_rewards=0,
  pseudo_reward=5,
  only_optimal_pseudo_rewards=False,
)

```


## Bonus

If you're a true pythonista who asks for forgiveness and not permission:

```python
from gym_spaceship_adventure import IllegalAction

action = env.action_space.sample()
try:
  state, reward, done, info = env.step(action)
except IllegalAction
  pass

```


## Notes

Python >= 3.6 recommended
