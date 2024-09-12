import torch 
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
    
def mask_tokens(tokens):
    # Find the [/INST] token position
    starts = []
    for batch in tokens:
        # 29914 +2 is end of the [/INST] token
        starts.append((batch == 29914).nonzero(as_tuple=True)[0].item()+2)

    # Mask the tokens
    range = torch.arange(tokens.shape[1]).expand(tokens.shape[0], -1)
    mask = range > torch.tensor(starts).unsqueeze(1)

    return mask.to('cuda')

def get_steer_like(model, tokenizer, data, steer_dir, steer_layer, coef):

    # Load the steer vector
    steer_vec = torch.load(steer_dir)
    steer_vec = steer_vec.to('cuda')

    # Define the hook function
    def steer_hook(module, input, output):
        return (output[0] + (steer_vec*coef), output[1])

    # Register the hook
    hook = model.model.layers[steer_layer].register_forward_hook(steer_hook)

    # Tokenize data
    inputs = tokenizer(data, return_tensors='pt', padding=True).to('cuda')

    likelihoods = []

    # Run the model while hooking with the steer vector
    for i in range(0, len(data), 20):

        # Print progress
        print(f'Generating {i} of {len(data)}')

        # Get the current batch
        sub_inputs = {k: v[i:i+20] for k, v in inputs.items()}

        # Generate sequences in batches
        with torch.no_grad():
            outputs = model(sub_inputs['input_ids'], attention_mask=sub_inputs['attention_mask'], labels=sub_inputs['input_ids'])

        # Calculate experimental log likelihood of continuation

        # Get the mask for the tokens
        unpadded_indices =  mask_tokens(sub_inputs['input_ids'])

        # Get the correct indices to meausre
        shift_unpadded_indices = unpadded_indices[..., 1:].contiguous()
        shift_labels = sub_inputs['input_ids'][..., 1:].contiguous()
        
        # Calculate the log likelihood
        exp_ll = outputs.logits[:,:-1].softmax(-1).log().gather(-1, shift_labels.unsqueeze(-1)).flatten(-2) * shift_unpadded_indices
        likelihoods += exp_ll.mean(-1).tolist()

    # Remove the hook
    hook.remove()

    return likelihoods


def main():

    # Load data and system prompts
    with open(cfg['data_dir'], 'r') as f:
        data = json.load(f)
    data = [ex['question'] for ex in data]


    # Load model and tokenizer
    model_name = cfg['model']
    model = AutoModelForCausalLM.from_pretrained(model_name).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'


    # Set the seed
    torch.manual_seed(cfg['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg['seed'])

    # Partition data into neg, pos
    partioned_data = [data[i:i+len(data)//2] for i in range(0, len(data), len(data)//2)]

    likelihoods = []
    unsteered_likelihoods = []

    # Get the likelihoods for each partition
    for part in partioned_data:

        # Get the steered likelihoods
        likelihoods+=get_steer_like(model, tokenizer, part, cfg['steer_dir'], cfg['steer_layer'], cfg['coef'])

        # Get the unsteered likelihoods
        unsteered_likelihoods+=get_steer_like(model, tokenizer, part, cfg['steer_dir'], cfg['steer_layer'], 0)

    # Save the likelihoods as toch tensors
    torch.save(torch.tensor(unsteered_likelihoods), cfg['output_dir'] + '_control.pt')
    torch.save(torch.tensor(likelihoods), cfg['output_dir'] + '_steer.pt')

cfg = {
    "model": 'meta-llama/Llama-2-7b-chat-hf',
    'steer_layer': 13,
    "coef": 2,
    'steer_dir': 'src/vecs/corrigible.pt',
    "data_dir": 'results/generated_data/corrigible.json',
    "output_dir": 'results/steeredness/corrigible',
    "seed": 42
}

if __name__ == '__main__':
    main()


