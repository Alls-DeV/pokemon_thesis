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

parser = argparse.ArgumentParser()

# Player arguments
parser.add_argument("--player_prompt_algo", default="io", choices=prompt_algos)
parser.add_argument("--player_backend", type=str, default="deepseek-v4-flash", choices=[
    # OpenAI models
    "gpt-5.4-nano", "gpt-4o", "gpt-4o-2024-05-13", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo",
    # Anthropic models
    "anthropic/claude-3.5-sonnet", "anthropic/claude-3-opus", "anthropic/claude-3-haiku",
    # Google models
    "google/gemini-pro", "gemini-2.0-flash", "gemini-2.0-pro", "gemini-2.0-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro",
    # Meta models
    "meta-llama/llama-3.1-70b-instruct", "meta-llama/llama-3.1-8b-instruct",
    # Mistral models
    "mistralai/mistral-7b-instruct", "mistralai/mixtral-8x7b-instruct",
    # Cohere models
    "cohere/command-r-plus", "cohere/command-r",
    # Perplexity models
    "perplexity/llama-3.1-sonar-small-128k", "perplexity/llama-3.1-sonar-large-128k",
    # DeepSeek models
    "deepseek-v4-flash", "deepseek-v4-pro", "deepseek-ai/deepseek-coder-33b-instruct", "deepseek-ai/deepseek-llm-67b-chat",
    # Microsoft models
    "microsoft/wizardlm-2-8x22b", "microsoft/phi-3-medium-128k-instruct",
    # Ollama models
    "ollama/gpt-oss:20b", "ollama/llama3.1:8b", "ollama/mistral", "ollama/qwen2.5", "ollama/gemma3:4b",
    # Local models (via OpenRouter)
    "llama", 'None' 
])
parser.add_argument("--player_name", type=str, default='polimi', choices=bot_choices)
parser.add_argument("--player_device", type=int, default=0)

# Opponent arguments
parser.add_argument("--opponent_prompt_algo", default="io", choices=prompt_algos)
parser.add_argument("--opponent_backend", type=str, default="gpt-5.4-nano", choices=[
    # OpenAI models
    "gpt-5.4-nano", "gpt-4o", "gpt-4o-2024-05-13", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo",
    # Anthropic models
    "anthropic/claude-3.5-sonnet", "anthropic/claude-3-opus", "anthropic/claude-3-haiku",
    # Google models
    "google/gemini-pro", "gemini-2.0-flash", "gemini-2.0-pro", "gemini-2.0-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro",
    # Ollama models
    "ollama/gpt-oss:20b", "ollama/llama3.1:8b", "ollama/mistral", "ollama/qwen2.5",
    # Meta models
    "meta-llama/llama-3.1-70b-instruct", "meta-llama/llama-3.1-8b-instruct",
    # Mistral models
    "mistralai/mistral-7b-instruct", "mistralai/mixtral-8x7b-instruct",
    # Cohere models
    "cohere/command-r-plus", "cohere/command-r",
    # Perplexity models
    "perplexity/llama-3.1-sonar-small-128k", "perplexity/llama-3.1-sonar-large-128k",
    # DeepSeek models
    "deepseek-v4-flash", "deepseek-v4-pro", "deepseek-ai/deepseek-coder-33b-instruct", "deepseek-ai/deepseek-llm-67b-chat",
    # Microsoft models
    "microsoft/wizardlm-2-8x22b", "microsoft/phi-3-medium-128k-instruct",
    # Local models (via OpenRouter)
    "llama", 'None'
])
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

    player_team_path = f"polimi/teams/team1.txt"
    with open(player_team_path, 'r') as f:
        player_team_data = f.read()
    player_team = Teambuilder.parse_showdown_team(player_team_data)
    for mon in player_team:
        if mon.species is not None:
            mon.nickname = mon.species
    player.update_team(Teambuilder.join_team(player_team))
    
    opponent_team_path = f"polimi/teams/team2.txt"
    with open(opponent_team_path, 'r') as f:
        opponent_team_data = f.read()
    opponent_team = Teambuilder.parse_showdown_team(opponent_team_data)
    for mon in opponent_team:
        if mon.species is not None:
            mon.nickname = mon.species
    opponent.update_team(Teambuilder.join_team(opponent_team))
    # opponent_teamloader = get_metamon_teams(args.battle_format, "modern_replays")
    # opponent.update_team(opponent_teamloader.yield_team())

    N = args.N
    for i in range(N):
        await player.battle_against(opponent, n_battles=1)


if __name__ == "__main__":
    asyncio.run(main())