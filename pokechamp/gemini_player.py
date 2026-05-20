from google import genai
from google.genai import types
from time import sleep
import os, sys
import json

class GeminiPlayer():
    def __init__(self, api_key=""):
        if api_key == "":
            self.api_key = os.getenv('GEMINI_API_KEY')
        else:
            self.api_key = api_key
        
        # Configure the Gemini API
        self.client = genai.Client(api_key=self.api_key)
        
        self.completion_tokens = 0
        self.prompt_tokens = 0
        
        # Map common model names to official API names
        self.model_mapping = {
            # Default to latest Gemini 2.5
            'gemini-flash': 'gemini-2.5-flash',
            'gemini-flash-2.5': 'gemini-2.5-flash',
            'gemini-pro': 'gemini-2.5-pro',
            'gemini-pro-2.5': 'gemini-2.5-pro',
            
            # Gemini 2.0 models
            'gemini-2.0-flash': 'gemini-2.0-flash',
            'gemini-2.0-flash-lite': 'gemini-2.0-flash-lite-preview-02-05',
            'gemini-2.0-pro': 'gemini-2.0-pro-exp-02-05',
            'gemini-2.0-pro-experimental': 'gemini-2.0-pro-exp-02-05',
            'gemini-2.0-flash-thinking': 'gemini-2.0-flash-thinking-exp-01-21',
            
            # Gemini 1.5 models (legacy support)
            'gemini-1.5-flash': 'gemini-1.5-flash',
            'gemini-1.5-pro': 'gemini-1.5-pro',
        }

    def get_LLM_action(self, system_prompt, user_prompt, model='gemini-2.5-flash', temperature=0.7, json_format=False, seed=None, stop=None, max_tokens=1000, actions=None, battle=None, ps_client=None, retries=3) -> tuple:
        if stop is None:
            stop = []
            
        try:
            # Map model name to official API name
            api_model_name = self.model_mapping.get(model, model)
            
            # Use proper configuration for Google GenAI SDK
            config_kwargs = {
                "system_instruction": system_prompt,
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
            if stop:
                config_kwargs["stop_sequences"] = stop
            if json_format:
                config_kwargs["response_mime_type"] = "application/json"
                
            config = types.GenerateContentConfig(**config_kwargs)
            
            # Generate response
            response = self.client.models.generate_content(
                model=api_model_name, 
                contents=user_prompt,
                config=config
            )
            
            # Extract text from response
            outputs = response.text
            
            # Token counting
            if response.usage_metadata:
                self.completion_tokens += getattr(response.usage_metadata, 'candidates_token_count', 0)
                self.prompt_tokens += getattr(response.usage_metadata, 'prompt_token_count', 0)
            else:
                self.completion_tokens += len(outputs.split()) * 1.3
                self.prompt_tokens += len(f"{system_prompt}\n\n{user_prompt}".split()) * 1.3
            
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
                    # Validate JSON
                    json.loads(json_str)
                    return json_str, True, outputs
                except json.JSONDecodeError:
                    # If JSON is invalid, attempt fallback matching
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
            
        except Exception as e:
            print(f'Gemini API error (get_LLM_action): {e}')
            if retries > 0:
                print(f"Retrying... ({retries} retries left)")
                sleep(2)
                return self.get_LLM_action(system_prompt, user_prompt, model, temperature, json_format, seed, stop, max_tokens, actions, battle, ps_client, retries - 1)
            else:
                print("Max retries exceeded.")
                sys.exit(1)
    
    def get_LLM_query(self, system_prompt, user_prompt, temperature=0.7, model='gemini-2.0-flash', json_format=False, seed=None, stop=None, max_tokens=1000, retries=3):
        if stop is None:
            stop = []
            
        try:
            api_model_name = self.model_mapping.get(model, model)
            
            # Use proper configuration for Google GenAI SDK
            config_kwargs = {
                "system_instruction": system_prompt,
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
            if stop:
                config_kwargs["stop_sequences"] = stop
            if json_format:
                config_kwargs["response_mime_type"] = "application/json"
                
            config = types.GenerateContentConfig(**config_kwargs)
            
            # Generate response
            response = self.client.models.generate_content(
                model=api_model_name, 
                contents=user_prompt,
                config=config
            )
            
            # Extract text from response
            message = response.text
            
            # Token counting
            if response.usage_metadata:
                self.completion_tokens += getattr(response.usage_metadata, 'candidates_token_count', 0)
                self.prompt_tokens += getattr(response.usage_metadata, 'prompt_token_count', 0)
            else:
                self.completion_tokens += len(message.split()) * 1.3
                self.prompt_tokens += len(f"{system_prompt}\n\n{user_prompt}".split()) * 1.3
                
        except Exception as e:
            print(f'Gemini API error (get_LLM_query): {e}')
            if retries > 0:
                print(f"Retrying... ({retries} retries left)")
                sleep(2)
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
