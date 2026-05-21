import asyncio
import os
import sys
import argparse
from poke_env.teambuilder import Teambuilder

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from common import *
from poke_env.player.team_util import TeamSet, get_llm_player

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
    num_teams = 5
    score_matrix = {p_idx: {o_idx: 0 for o_idx in range(1, num_teams + 1)} for p_idx in range(1, num_teams + 1)}
    
    # Initialize players once outside the loop to avoid name collisions
    player = get_llm_player(args, 
                            args.player_backend, 
                            args.player_prompt_algo, 
                            args.player_name, 
                            device=args.player_device,
                            PNUMBER1=PNUMBER1,  # for name uniqueness locally
                            battle_format=args.battle_format,
                            team_idx=1) # Initial dummy value, will be updated
    
    opponent = get_llm_player(args, 
                            args.opponent_backend, 
                            args.opponent_prompt_algo, 
                            args.opponent_name, 
                            device=args.opponent_device,
                            PNUMBER1=PNUMBER1 + '2',  # for name uniqueness locally
                            battle_format=args.battle_format,
                            team_idx=1) # Initial dummy value, will be updated

    for _ in range(args.N):
        for player_team_idx in range(1, num_teams + 1):
            for opponent_team_idx in range(1, num_teams + 1):
                
                # Update team_idx for strategies if the bots support it
                if hasattr(player, 'set_team_idx'):
                    player.set_team_idx(player_team_idx)
                elif hasattr(player, 'team_idx'):
                    player.team_idx = player_team_idx
                    
                if hasattr(opponent, 'set_team_idx'):
                    opponent.set_team_idx(opponent_team_idx)
                elif hasattr(opponent, 'team_idx'):
                    opponent.team_idx = opponent_team_idx

                player_team_path = f"polimi/teams/team{player_team_idx}.txt"
                opponent_team_path = f"polimi/teams/team{opponent_team_idx}.txt"
                load_team(player_team_path, player)
                load_team(opponent_team_path, opponent)

                wins_before = player.n_won_battles
                print(f"Starting battle {player_team_idx} vs {opponent_team_idx}")
                await player.battle_against(opponent, n_battles=1)
                wins_after = player.n_won_battles
                
                if wins_after > wins_before:
                    score_matrix[player_team_idx][opponent_team_idx] += 1
                    print(f"Won battle {player_team_idx} vs {opponent_team_idx}")
                else:
                    print(f"Lost battle {player_team_idx} vs {opponent_team_idx}")

    # Print the score matrix
    print("\n" + "="*50)
    print(f"BATTLE RESULTS MATRIX (Player Win Rate over {args.N} battles)")
    print("="*50)
    print("Row: Player Team | Col: Opponent Team")
    print("-" * 50)
    
    header = "P \\ O |" + "".join([f" Team {i} |" for i in range(1, num_teams + 1)])
    print(header)
    print("-" * len(header))
    
    for p_idx in range(1, num_teams + 1):
        row_str = f"Team {p_idx} |"
        for o_idx in range(1, num_teams + 1):
            wins = score_matrix[p_idx][o_idx]
            win_rate = (wins / args.N) * 100
            win_rate_str = f"{win_rate:.0f}%"
            row_str += f" {win_rate_str:^6} |"
        print(row_str)
        print("-" * len(header))

if __name__ == "__main__":
    asyncio.run(main())
