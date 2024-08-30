import torch 
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer

def gen_cont(model, tokenizer, data, sys_prompts):
    # Set model to eval mode
    model.eval()

    # Add System Prompt Tokens to Data
    data = ["[INST] <<SYS>>\n" + sys+ "\n<</SYS>>\n\n"+ ex + " [/INST]" for sys in sys_prompts for ex in data]

    # Tokenize data
    data = tokenizer(data, return_tensors='pt', padding=True).to('cuda')
    
    # Generate sequence
    with torch.no_grad():
        outputs = model.generate(data, max_length=100, pad_token_id=tokenizer.eos_token_id)

    
    # Decode sequence
    generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    return generated_texts


def main():
    # Load data and system prompts
    with open(cfg.data_dir, 'r') as f:
        data = f.readlines()

    sys_prompts = cfg.sys_prompts

    # Load model and tokenizer
    model_name = cfg.model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Generater sentences
    outputs = gen_cont(model, tokenizer, data, sys_prompts)

    # Write to file
    with open(cfg.output_dir, 'w') as f:
        f.write('\n'.join(outputs))


if __name__ == '__main__':
    main()


@dataclass
class config:
    model: str = 'meta-llama/Llama-2-7b-chat-hf'
    data_dir: str = 'src/data/myopic.txt'
    output_dir: str = 'results/generated_data/myopic.txt'
    sys_prompts: list = ['','You are a helpful AI assistant.','']
    
cfg = config()
