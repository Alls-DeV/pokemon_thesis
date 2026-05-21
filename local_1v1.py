import asyncio
import os
import sys
import argparse
from poke_env.teambuilder import Teambuilder

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from common import *
from poke_env.player.team_util import TeamSet, get_llm_player, get_metamon_teams, load_random_team

def load_team(path, player=None):
    with open(path, 'r') as f:
        team_data = f.read()
    team = Teambuilder.parse_showdown_team(team_data)
    for mon in team:
        if mon.species is not None:
            mon.nickname = mon.species
    joined_team = Teambuilder.join_team(team)
    if player is not None:
        player.update_team(joined_team)
    return joined_team

parser = argparse.ArgumentParser()

# Player arguments
parser.add_argument("--player_prompt_algo", default="io", choices=prompt_algos)
parser.add_argument("--player_backend", type=str, default="deepseek-v4-pro", choices=AVAILABLE_MODELS)
parser.add_argument("--player_name", type=str, default='polimi', choices=bot_choices)
parser.add_argument("--player_device", type=int, default=0)

# Opponent arguments
parser.add_argument("--opponent_prompt_algo", default="io", choices=prompt_algos)
parser.add_argument("--opponent_backend", type=str, default="gpt-5.4-nano", choices=AVAILABLE_MODELS)
parser.add_argument("--opponent_name", type=str, default='pokellmon', choices=bot_choices)
parser.add_argument("--opponent_device", type=int, default=0)

# Shared arguments
parser.add_argument("--temperature", type=float, default=0.3)
parser.add_argument("--battle_format", default="gen9ou", choices=["gen8randombattle", "gen8ou", "gen9ou", "gen9randombattle", "gen9vgc2024regg"])
parser.add_argument("--log_dir", type=str, default="./battle_log/one_vs_one")
parser.add_argument("--N", type=int, default=1)

args = parser.parse_args()

async def main():
    player = get_llm_player(args, 
                            args.player_backend, 
                            args.player_prompt_algo, 
                            args.player_name, 
                            device=args.player_device,
                            PNUMBER1=PNUMBER1,  # for name uniqueness locally
                            battle_format=args.battle_format)
    
    opponent = get_llm_player(args, 
                            args.opponent_backend, 
                            args.opponent_prompt_algo, 
                            args.opponent_name, 
                            device=args.opponent_device,
                            PNUMBER1=PNUMBER1 + '2',  # for name uniqueness locally
                            battle_format=args.battle_format)

    player_team_path = f"polimi/teams/team2.txt"
    load_team(player_team_path, player)
    
    opponent_team_path = f"polimi/teams/team1.txt"
    load_team(opponent_team_path, opponent)
    # opponent_teamloader = get_metamon_teams(args.battle_format, "modern_replays")
    # opponent.update_team(opponent_teamloader.yield_team())

    N = args.N
    for i in range(N):
        await player.battle_against(opponent, n_battles=1)


if __name__ == "__main__":
    asyncio.run(main())
