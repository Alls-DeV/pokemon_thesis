import json
import re
import sys
import os

def parse_smogon_stats(filepath, output_filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f]

    pokemon_data = {}
    current_pokemon = None
    current_section = None
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        if not line:
            i += 1
            continue
            
        if line.startswith('+--'):
            # Check if the next line is also a separator
            if i + 1 < len(lines) and lines[i+1].startswith('+--'):
                # End of current pokemon, start of next
                current_pokemon = None
                current_section = None
                i += 2
                continue
            i += 1
            continue
            
        if line.startswith('|'):
            content = line.strip('| \t')
        else:
            i += 1
            continue
            
        if not content:
            i += 1
            continue
            
        if current_pokemon is None:
            current_pokemon = content
            pokemon_data[current_pokemon] = {
                'metadata': {},
                'Abilities': {},
                'Items': {},
                'Spreads': {},
                'Moves': {},
                'Tera Types': {},
                'Teammates': {}
            }
            current_section = 'metadata'
            i += 1
            continue
            
        if content in ['Abilities', 'Items', 'Spreads', 'Moves', 'Tera Types', 'Teammates', 'Checks and Counters']:
            current_section = content
            i += 1
            continue
            
        if current_section == 'Checks and Counters':
            i += 1
            continue
            
        if current_section == 'metadata':
            if ':' in content:
                key, val = content.split(':', 1)
                try:
                    val_num = float(val.strip())
                    pokemon_data[current_pokemon]['metadata'][key.strip()] = val_num
                except ValueError:
                    pokemon_data[current_pokemon]['metadata'][key.strip()] = val.strip()
                    
        elif current_section in ['Abilities', 'Items', 'Spreads', 'Moves', 'Tera Types', 'Teammates']:
            match = re.search(r'^(.*?) (\d+(?:\.\d+)?)%$', content)
            if match:
                name = match.group(1).strip()
                percentage = float(match.group(2))
                pokemon_data[current_pokemon][current_section][name] = percentage
                
        i += 1

    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(pokemon_data, f, indent=4)
        
    print(f"Parsed {len(pokemon_data)} Pokemon data into {output_filepath}")

if __name__ == '__main__':
    input_file = sys.argv[1] if len(sys.argv) > 1 else 'polimi/smogon_stats.txt'
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'polimi/smogon_stats.json'
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        sys.exit(1)
    parse_smogon_stats(input_file, output_file)
