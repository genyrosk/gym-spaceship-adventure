from gym.envs.registration import register

register(
    id="spaceship-adventure-v0",
    entry_point="gym_spaceship_adventure.envs:SpaceshipAdventureEnv",
)
