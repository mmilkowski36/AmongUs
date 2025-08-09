## experiment_config
Some values in experiment_config can be overriden by explicitly stating values in command line
Default values are specified here:
number_of_runs = 1
model = "RANDOM" //in progress
number_of_agents = 3

## game_config: 
THREE_MEMBER_GAME = {
    "num_players": 3,
    "num_impostors": 1,
    "num_common_tasks": 1,
    "num_short_tasks": 1,
    "num_long_tasks": 0,
    "discussion_rounds": 3,
    "max_num_buttons": 2,
    "kill_cooldown": 1,
    "max_timesteps": 5,
}

FIVE_MEMBER_GAME = {
    "num_players": 5,
    "num_impostors": 1,
    "num_common_tasks": 1,
    "num_short_tasks": 1,
    "num_long_tasks": 0,
    "discussion_rounds": 3,
    "max_num_buttons": 2,
    "kill_cooldown": 3,
    "max_timesteps": 20,
}

SEVEN_MEMBER_GAME = {
    "num_players": 7,
    "num_impostors": 2,
    "num_common_tasks": 1,
    "num_short_tasks": 1,
    "num_long_tasks": 1,
    "discussion_rounds": 3,
    "max_num_buttons": 2,
    "kill_cooldown": 3,
    "max_timesteps": 50,
}