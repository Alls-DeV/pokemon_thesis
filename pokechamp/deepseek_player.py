import os
import sys
import json
from time import sleep
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

    def get_LLM_action(self, system_prompt, user_prompt, model='deepseek-v4-flash', temperature=0.7, json_format=False, seed=None, stop=None, max_tokens=8000, actions=None, battle=None, ps_client=None, retries=3) -> tuple:
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
            
            if stop:
                kwargs["stop"] = stop
                
            if json_format:
                # DeepSeek officially supports JSON output for `deepseek-chat` model
                kwargs["response_format"] = {"type": "json_object"}
            
            # Use max_tokens for DeepSeek as they might not support max_completion_tokens fully yet
            kwargs["max_tokens"] = max_tokens

            response = self.client.chat.completions.create(**kwargs)
            
            outputs = response.choices[0].message.content
            
            # log completion tokens
            if response.usage:
                self.completion_tokens += getattr(response.usage, 'completion_tokens', 0)
                self.prompt_tokens += getattr(response.usage, 'prompt_tokens', 0)
                
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
                
                try:
                    json.loads(json_str)
                    return json_str, True, outputs
                except json.JSONDecodeError:
                    start_idx = outputs.find('{')
                    end_idx = outputs.rfind('}')
                    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                        fallback_json = outputs[start_idx:end_idx + 1]
                        try:
                            json.loads(fallback_json)
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
    
    def get_LLM_query(self, system_prompt, user_prompt, temperature=0.7, model='deepseek-v4-flash', json_format=False, seed=None, stop=None, max_tokens=2000, retries=3):
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
                return self.get_LLM_query(system_prompt, user_prompt, temperature, model, json_format, seed, stop, max_tokens, retries - 1)
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