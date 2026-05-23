import os
import sys
import json
import atexit
from time import sleep, time
from openai import OpenAI, OpenAIError

class GPTPlayer():
    def __init__(self, api_key=""):
        if api_key == "":
            self.api_key = os.getenv('OPENAI_API_KEY')
        else:
            self.api_key = api_key
        
        self.client = OpenAI(api_key=self.api_key)
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.game_stats = {}
        self.turn_stats = {}
        atexit.register(self.log_game_stats)

    def log_game_stats(self):
        import csv
        log_file = "openai_game_stats.csv"
        file_exists = os.path.isfile(log_file)
        
        if getattr(self, 'is_polimi', False):
            with open(log_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                # Write header if file is new
                if not file_exists:
                    writer.writerow(["battle_id", "prompt_type", "total_requests", "mean_time_seconds", "mean_tokens", "switch_skips", "merger_switches", "merger_moves"])
                
                for b_id, stats in self.game_stats.items():
                    for p_type in ["merger", "move", "switch"]:
                        times = stats[p_type]["times"]
                        tokens = stats[p_type]["tokens"]
                        
                        mean_time = sum(times) / len(times) if times else 0.0
                        mean_tokens = sum(tokens) / len(tokens) if tokens else 0.0
                        skips = stats["switch"]["skips"] if p_type == "switch" else 0
                        merger_switches = stats["merger"]["switches"] if p_type == "merger" else 0
                        merger_moves = stats["merger"]["moves"] if p_type == "merger" else 0
                        
                        writer.writerow([b_id, p_type, len(times), f"{mean_time:.2f}", f"{mean_tokens:.2f}", skips, merger_switches, merger_moves])
                        
            print(f"Logged OpenAI game stats to {log_file}")

        # Log turn stats
        for b_id, turns in self.turn_stats.items():
            log_dir = "battle_log/turn_stats"
            os.makedirs(log_dir, exist_ok=True)
            file_path = os.path.join(log_dir, f"OpenAI_{b_id}_turn_stats.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                total_prompt = 0
                total_completion = 0
                total_time = 0.0
                num_turns = len(turns)
                
                for turn, stats in sorted(turns.items()):
                    p_tks = stats["prompt_tokens"]
                    c_tks = stats["completion_tokens"]
                    t_time = stats["time"]
                    f.write(f"Turn {turn}:\n")
                    f.write(f"Input Tokens: {p_tks}\n")
                    f.write(f"Output Tokens: {c_tks}\n")
                    f.write(f"Time Spent: {t_time:.2f} seconds\n\n")
                    
                    total_prompt += p_tks
                    total_completion += c_tks
                    total_time += t_time
                    
                if num_turns > 0:
                    f.write("--- Mean over all turns ---\n")
                    f.write(f"Mean Input Tokens: {total_prompt / num_turns:.2f}\n")
                    f.write(f"Mean Output Tokens: {total_completion / num_turns:.2f}\n")
                    f.write(f"Mean Time Spent: {total_time / num_turns:.2f} seconds\n")
        print(f"Logged OpenAI turn stats to battle_log/turn_stats/")

    def get_LLM_action(self, system_prompt, user_prompt, model, temperature=0.7, json_format=False, seed=None, stop=None, max_tokens=20000, actions=None, battle=None, ps_client=None, retries=3) -> tuple:
        if stop is None:
            stop = []
            
        try:
            is_reasoning = model.startswith('o1') or model.startswith('o3')
            
            kwargs = {
                "model": model,
                "messages": []
            }
            
            # Reasoning models like o1/o3 often require all context in developer or user roles,
            # and do not support 'temperature' or 'stop' sequences natively on all endpoints.
            if is_reasoning:
                kwargs["messages"].append({"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"})
            else:
                kwargs["messages"].append({"role": "system", "content": system_prompt})
                kwargs["messages"].append({"role": "user", "content": user_prompt})
                kwargs["temperature"] = temperature
                if stop:
                    kwargs["stop"] = stop
                if json_format:
                    kwargs["response_format"] = {"type": "json_object"}
            
            # max_completion_tokens is the standard for both modern reasoning models and standard chat
            kwargs["max_completion_tokens"] = max_tokens

            prompt_type = "unknown"
            if "You have two possible actions to choose from:" in user_prompt:
                prompt_type = "merger"
            elif "Available switches:" in user_prompt:
                prompt_type = "switch"
            elif "Available moves:" in user_prompt:
                prompt_type = "move"

            start_time = time()
            response = self.client.chat.completions.create(**kwargs)
            elapsed_time = time() - start_time
            outputs = response.choices[0].message.content
            
            b_tag = battle.battle_tag if battle and hasattr(battle, 'battle_tag') else "default"
            p_id = battle.player_username if battle and hasattr(battle, 'player_username') else "agent"
            p_id = "".join(c for c in p_id if c.isalnum() or c in ('_', '-'))
            b_id = f"{b_tag}_{p_id}"
            
            log_dir = "battle_log/openai_prompts"
            os.makedirs(log_dir, exist_ok=True)
            match_log_file = os.path.join(log_dir, f"prompts_{b_id}.log")
            
            turn_info = battle.turn if battle and hasattr(battle, 'turn') else "Unknown"
            
            with open(match_log_file, "a", encoding="utf-8") as f:
                f.write(f"\n\n\n=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~\n")
                f.write(f"TURN: {turn_info} | PROMPT TYPE: {prompt_type}\n")
                f.write(f"SYSTEM PROMPT:\n{system_prompt}\n")
                f.write(f"\nUSER PROMPT:\n{user_prompt}\n")
                f.write(f"\nOUTPUT:\n{outputs}\n")
            
            # log completion tokens
            cur_completion_tks = 0
            cur_prompt_tks = 0
            if response.usage:
                cur_completion_tks = getattr(response.usage, 'completion_tokens', 0)
                cur_prompt_tks = getattr(response.usage, 'prompt_tokens', 0)
                self.completion_tokens += cur_completion_tks
                self.prompt_tokens += cur_prompt_tks
                
            if b_id not in self.game_stats:
                self.game_stats[b_id] = {
                    "move": {"times": [], "tokens": []},
                    "switch": {"times": [], "tokens": [], "skips": 0},
                    "merger": {"times": [], "tokens": [], "switches": 0, "moves": 0}
                }
            
            if prompt_type in self.game_stats[b_id]:
                self.game_stats[b_id][prompt_type]["times"].append(elapsed_time)
                self.game_stats[b_id][prompt_type]["tokens"].append(cur_prompt_tks + cur_completion_tks)

            # Log turn stats
            if battle and hasattr(battle, 'turn'):
                turn = battle.turn
                if b_id not in self.turn_stats:
                    self.turn_stats[b_id] = {}
                if turn not in self.turn_stats[b_id]:
                    self.turn_stats[b_id][turn] = {"prompt_tokens": 0, "completion_tokens": 0, "time": 0.0}
                
                self.turn_stats[b_id][turn]["prompt_tokens"] += cur_prompt_tks
                self.turn_stats[b_id][turn]["completion_tokens"] += cur_completion_tks
                self.turn_stats[b_id][turn]["time"] += elapsed_time

            if json_format:
                # Cleanup potential markdown formatting
                json_str = outputs.strip()
                if json_str.startswith("```json"):
                    json_str = json_str[7:]
                elif json_str.startswith("```"):
                    json_str = json_str[3:]
                if json_str.endswith("```"):
                    json_str = json_str[:-3]
                json_str = json_str.strip()
                
                parsed_json = None
                try:
                    parsed_json = json.loads(json_str)
                    
                    if prompt_type == "switch":
                        if str(parsed_json.get("switch", "")).strip().lower() == "nothing":
                            self.game_stats[b_id]["switch"]["skips"] += 1
                    elif prompt_type == "merger":
                        choice = str(parsed_json.get("choice", "")).strip().lower()
                        if choice == "switch":
                            self.game_stats[b_id]["merger"]["switches"] += 1
                        elif choice == "move":
                            self.game_stats[b_id]["merger"]["moves"] += 1
                            
                    return json_str, True, outputs
                except json.JSONDecodeError:
                    start_idx = outputs.find('{')
                    end_idx = outputs.rfind('}')
                    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                        fallback_json = outputs[start_idx:end_idx + 1]
                        try:
                            parsed_json = json.loads(fallback_json)
                            
                            if prompt_type == "switch":
                                if str(parsed_json.get("switch", "")).strip().lower() == "nothing":
                                    self.game_stats[b_id]["switch"]["skips"] += 1
                            elif prompt_type == "merger":
                                choice = str(parsed_json.get("choice", "")).strip().lower()
                                if choice == "switch":
                                    self.game_stats[b_id]["merger"]["switches"] += 1
                                elif choice == "move":
                                    self.game_stats[b_id]["merger"]["moves"] += 1
                                    
                            return fallback_json, True, outputs
                        except:
                            pass
                    return outputs, True, outputs
            
            return outputs, False, outputs

        except OpenAIError as e:
            print(f'OpenAI API error (get_LLM_action): {e}')
            if retries > 0:
                print(f"Retrying... ({retries} retries left)")
                sleep(5)
                return self.get_LLM_action(system_prompt, user_prompt, model, temperature, json_format, seed, stop, max_tokens, actions, battle, ps_client, retries - 1)
            else:
                print("Max retries exceeded.")
                sys.exit(1)
    
    def get_LLM_query(self, system_prompt, user_prompt, model, temperature=0.7, json_format=False, seed=None, stop=None, max_tokens=200, retries=3):
        if stop is None:
            stop = []
            
        try:
            is_reasoning = model.startswith('o1') or model.startswith('o3')
            
            kwargs = {
                "model": model,
                "messages": []
            }
            
            if is_reasoning:
                kwargs["messages"].append({"role": "user", "content": f"{system_prompt}\n\n{user_prompt}"})
            else:
                kwargs["messages"].append({"role": "system", "content": system_prompt})
                kwargs["messages"].append({"role": "user", "content": user_prompt})
                kwargs["temperature"] = temperature
                if stop:
                    kwargs["stop"] = stop
                if json_format:
                    kwargs["response_format"] = {"type": "json_object"}
                    
            kwargs["max_completion_tokens"] = max_tokens
            
            response = self.client.chat.completions.create(**kwargs)
            message = response.choices[0].message.content
            
            if response.usage:
                self.completion_tokens += getattr(response.usage, 'completion_tokens', 0)
                self.prompt_tokens += getattr(response.usage, 'prompt_tokens', 0)
                
        except OpenAIError as e:
            print(f'OpenAI API error (get_LLM_query): {e}')
            if retries > 0:
                print(f"Retrying... ({retries} retries left)")
                sleep(5)
                return self.get_LLM_query(system_prompt, user_prompt, model, temperature, json_format, seed, stop, max_tokens, retries - 1)
            else:
                print("Max retries exceeded.")
                sys.exit(1)
        
        if json_format:
            # Cleanup potential markdown formatting
            json_str = message.strip()
            if json_str.startswith("```json"):
                json_str = json_str[7:]
            elif json_str.startswith("```"):
                json_str = json_str[3:]
            if json_str.endswith("```"):
                json_str = json_str[:-3]
            json_str = json_str.strip()
            
            try:
                json.loads(json_str)
                return json_str, True
            except json.JSONDecodeError:
                start_idx = message.find('{')
                end_idx = message.rfind('}')
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    fallback_json = message[start_idx:end_idx + 1]
                    try:
                        json.loads(fallback_json)
                        return fallback_json, True
                    except:
                        pass
                return message, True
        
        return message, False
