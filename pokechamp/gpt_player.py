import os
import sys
import json
from time import sleep
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

    def get_LLM_action(self, system_prompt, user_prompt, model='gpt-5.4-mini', temperature=0.7, json_format=False, seed=None, stop=None, max_tokens=20000, actions=None, battle=None, ps_client=None, retries=3) -> tuple:
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
            print(f'OpenAI API error (get_LLM_action): {e}')
            if retries > 0:
                print(f"Retrying... ({retries} retries left)")
                sleep(5)
                return self.get_LLM_action(system_prompt, user_prompt, model, temperature, json_format, seed, stop, max_tokens, actions, battle, ps_client, retries - 1)
            else:
                print("Max retries exceeded.")
                sys.exit(1)
    
    def get_LLM_query(self, system_prompt, user_prompt, temperature=0.7, model='gpt-5.4-mini', json_format=False, seed=None, stop=None, max_tokens=200, retries=3):
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