import os
import sys
import json
import atexit
from time import sleep, time
from openai import OpenAI, OpenAIError

class DeepSeekPlayer():
    def __init__(self, api_key=""):
        if api_key == "":
            self.api_key = os.getenv('DEEPSEEK_API_KEY')
        else:
            self.api_key = api_key
        
        # DeepSeek uses an OpenAI-compatible API format, so we can use the OpenAI client
        # but point it to the DeepSeek base URL.
        self.client = OpenAI(
            api_key=self.api_key, 
            base_url="https://api.deepseek.com"
        )
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.game_stats = {}
        atexit.register(self.log_game_stats)

    def log_game_stats(self):
        import csv
        log_file = "deepseek_game_stats.csv"
        file_exists = os.path.isfile(log_file)
        
        with open(log_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Write header if file is new
            if not file_exists:
                writer.writerow(["battle_id", "prompt_type", "total_requests", "mean_time_seconds", "mean_tokens", "switch_skips"])
            
            for b_id, stats in self.game_stats.items():
                for p_type in ["merger", "move", "switch"]:
                    times = stats[p_type]["times"]
                    tokens = stats[p_type]["tokens"]
                    
                    mean_time = sum(times) / len(times) if times else 0.0
                    mean_tokens = sum(tokens) / len(tokens) if tokens else 0.0
                    skips = stats["switch"]["skips"] if p_type == "switch" else 0
                    
                    writer.writerow([b_id, p_type, len(times), f"{mean_time:.2f}", f"{mean_tokens:.2f}", skips])
                    
        print(f"Logged DeepSeek game stats to {log_file}")

    def get_LLM_action(self, system_prompt, user_prompt, model, temperature=0.7, json_format=False, seed=None, stop=None, max_tokens=8000, actions=None, battle=None, ps_client=None, retries=3) -> tuple:
        if stop is None:
            stop = []
            
        try:
            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            }
            
            kwargs["temperature"] = temperature
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
            
            if stop:
                kwargs["stop"] = stop
                
            if json_format:
                # DeepSeek officially supports JSON output for `deepseek-chat` model
                kwargs["response_format"] = {"type": "json_object"}
            
            # Use max_tokens for DeepSeek as they might not support max_completion_tokens fully yet
            kwargs["max_tokens"] = max_tokens

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
            
            b_id = battle.battle_tag if battle and hasattr(battle, 'battle_tag') else "default"
            log_dir = "battle_log/deepseek_prompts"
            os.makedirs(log_dir, exist_ok=True)
            match_log_file = os.path.join(log_dir, f"prompts_{b_id}.log")
            
            with open(match_log_file, "a", encoding="utf-8") as f:
                f.write(f"\n\n\n=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~\n")
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
            
            b_id = battle.battle_tag if battle and hasattr(battle, 'battle_tag') else "default"
            if b_id not in self.game_stats:
                self.game_stats[b_id] = {
                    "move": {"times": [], "tokens": []},
                    "switch": {"times": [], "tokens": [], "skips": 0},
                    "merger": {"times": [], "tokens": []}
                }
            
            if prompt_type in self.game_stats[b_id]:
                self.game_stats[b_id][prompt_type]["times"].append(elapsed_time)
                self.game_stats[b_id][prompt_type]["tokens"].append(cur_prompt_tks + cur_completion_tks)
            
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
                                    
                            return fallback_json, True, outputs
                        except:
                            pass
                    return outputs, True, outputs
            
            return outputs, False, outputs

        except OpenAIError as e:
            print(f'DeepSeek API error (get_LLM_action): {e}')
            if retries > 0:
                print(f"Retrying... ({retries} retries left)")
                sleep(5)
                return self.get_LLM_action(system_prompt, user_prompt, model, temperature, json_format, seed, stop, max_tokens, actions, battle, ps_client, retries - 1)
            else:
                print("Max retries exceeded.")
                sys.exit(1)
    
    def get_LLM_query(self, system_prompt, user_prompt, model, temperature=0.7, json_format=False, seed=None, stop=None, max_tokens=2000, retries=3):
        if stop is None:
            stop = []
            
        try:
            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            }
            
            kwargs["temperature"] = temperature
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
            
            if stop:
                kwargs["stop"] = stop
                
            if json_format:
                kwargs["response_format"] = {"type": "json_object"}
                    
            kwargs["max_tokens"] = max_tokens
            
            response = self.client.chat.completions.create(**kwargs)
            message = response.choices[0].message.content
            
            if response.usage:
                self.completion_tokens += getattr(response.usage, 'completion_tokens', 0)
                self.prompt_tokens += getattr(response.usage, 'prompt_tokens', 0)
                
        except OpenAIError as e:
            print(f'DeepSeek API error (get_LLM_query): {e}')
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