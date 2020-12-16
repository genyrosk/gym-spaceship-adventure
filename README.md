# Spacehip Adventure environment for OpenAI gym 🚀

A cool variation on the [frozen-lake](https://gym.openai.com/envs/FrozenLake-v0/) environment.

The agent controls the movement of a spaceship in a grid world.
Some tiles of the grid are navigable, and others lead to the agent colliding with an asteroid 💥.
Additionally, the agent is awarded a one-time boost ⚡ that can move it accross 4 tiles in one go.
The agent is rewarded for finding a navigable path to a goal tile.

The space is described using a grid like the following:

```

  AA-A-G
  --A---         (G: goal tile)
  ----AA         (-: empty space, safe)
  --A--A         (A: asteroid, collide to your doom)
  A--A--         (S: starting point, safe)
  S-----

```

The episode ends when you reach the goal or collide with an asteroid.
You receive a reward of -1 for each movement, 10 if you reach the goal, and zero if you collide.


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
env = gym.make('spaceship-adventure-v0')

# Randomly generated map of a certain size and probability of empty tile `p`
env = gym.make('spaceship-adventure-v0', use_default_map=False, size=6, p=0.7)

# get available actions
possible_actions = env.get_possible_actions()

# take a step
state, reward, done, info = env.step(3)

# render the current state
env.render()

# reset
env.reset()

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
