# This file converts a json file in (Rimsky et al format) into a txt file
import json
from dataclasses import dataclass


def json_to_txt(json_file, txt_file):  
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    with open(txt_file, 'w') as f:
        for d in data:
            f.write(d['question'] + '\n')
            
    return txt_file

@dataclass
class config:
    json_file: str = 'src/data/myopia_open.json'
    txt_file: str = 'src/data/myopia_open.txt'

cfg = config()
if __name__ == '__main__':
    json_to_txt(cfg.json_file, cfg.txt_file) 
