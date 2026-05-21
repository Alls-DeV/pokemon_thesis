import random
import numpy as np
import os
import importlib
import inspect

prompt_algos = [
    "io", 
    "sc", 
    "cot", 
    "tot", 
    "minimax", 
    "heuristic", 
    'max_power',
    'one_step',
    'random',
    'mcp'
    ]

def get_available_bots():
    """Get a list of all available bot names from the bots folder."""
    bot_names = []
    bots_dir = os.path.join(os.path.dirname(__file__), 'bots')
    
    if not os.path.exists(bots_dir):
        return bot_names
    
    # Look for Python files in the bots directory
    for filename in os.listdir(bots_dir):
        if filename.endswith('.py') and not filename.startswith('__'):
            # Remove .py extension and _bot suffix to get the bot name
            bot_name = filename[:-3]  # Remove .py
            if bot_name.endswith('_bot'):
                bot_name = bot_name[:-4]  # Remove _bot suffix
            bot_names.append(bot_name)
    
    return bot_names

# Get available bot names from the bots folder
available_bots = get_available_bots()

# Combine built-in bots with custom bots
bot_choices = ['pokechamp', 'pokellmon', 'one_step', 'abyssal', 'max_power', 'random', 'vgc', 'polimi'] + available_bots

PNUMBER1 = str(np.random.randint(0,10000))
print(PNUMBER1)
seed = 100
random.seed(seed)
np.random.seed(seed)

AVAILABLE_MODELS = [
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
]