import torch 
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
    
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

        # Calculate experimental log likelihood
        unpadded_indices = (sub_inputs['input_ids'] != tokenizer.pad_token_id)

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

    # Partition data into neg, neutral, and pos
    partioned_data = [data[i:i+len(data)//3] for i in range(0, len(data), len(data)//3)]

    likelihoods = []
    unsteered_likelihoods = []

    # Get the likelihoods for each partition
    for part in partioned_data:

        # Get the steered likelihoods
        likelihoods+=get_steer_like(model, tokenizer, part, cfg['steer_dir'], cfg['steer_layer'], cfg['coef'])

        # Get the unsteered likelihoods
        unsteered_likelihoods+=get_steer_like(model, tokenizer, part, cfg['steer_dir'], cfg['steer_layer'], 0)

    # Format x-axis
    x_axis = np.concatenate([np.zeros(len(data)//3), np.ones(len(data)//3), np.full(len(data)//3, 2)])

    # Plot the likelihoods
    plt.figure()
    plt.scatter(x_axis, likelihoods, label='Steered Model', color='red')
    plt.scatter(x_axis, unsteered_likelihoods, label='Unsteered Model', color='blue')
    plt.xlabel('Example Type')
    plt.ylabel('Log-Likelihood')
    plt.legend()
    plt.savefig(cfg['output_dir'] + '_scatter.png')

    # Plot the likelihoods with bin sorting

    #split control into 3 parts
    partioned_like_un = [unsteered_likelihoods[i:i+len(unsteered_likelihoods)//3] for i in range(0, len(unsteered_likelihoods), len(unsteered_likelihoods)//3)]

    # Get the sorted indices for the unsteered likelihoods for each partition
    sorted_indices = [np.argsort(part)+((len(likelihoods)//3)*i) for i,part in enumerate(partioned_like_un)]
    sorted_indices = np.concatenate(sorted_indices)

    # Sort the steered likelihoods
    sorted_like = [likelihoods[i] for i in sorted_indices]
    sorted_like_un = [unsteered_likelihoods[i] for i in sorted_indices]

    plt.figure()
    plt.ylim(-2.5, 0)
    plt.scatter(np.arange(len(sorted_like)), sorted_like, label='Steered Model', color='red')
    plt.scatter(np.arange(len(sorted_like_un)), sorted_like_un, label='Unsteered Model', color='blue')
    plt.xticks(visible = False) 


    plt.xlabel('Input_ID')
    plt.ylabel('Log-Likelihood')
    plt.legend()
    plt.savefig(cfg['output_dir'] + '_bin_sorted.png')


    # Plot the likelihoods with total sorting

    sorted_indices = np.argsort(unsteered_likelihoods)

    # Sort the steered likelihoods
    sorted_like = [likelihoods[i] for i in sorted_indices]
    sorted_like_un = [unsteered_likelihoods[i] for i in sorted_indices]

    plt.figure()
    plt.ylim(-2.5, 0)
    plt.scatter(np.arange(len(sorted_like)), sorted_like, label='Steered Model', color='red')
    plt.scatter(np.arange(len(sorted_like_un)), sorted_like_un, label='Unsteered Model', color='blue')
    plt.xticks(visible = False) 


    plt.xlabel('Input_ID')
    plt.ylabel('Log-Likelihood')
    plt.legend()
    plt.savefig(cfg['output_dir'] + '_total_sorted.png')

cfg = {
    "model": 'meta-llama/Llama-2-7b-chat-hf',
    'steer_layer': 13,
    "coef": 2,
    'steer_dir': 'src/vecs/myopia.pt',
    "data_dir": 'results/generated_data/myopia.json',
    "output_dir": 'results/steeredness/myopia',
    "seed": 42
}

if __name__ == '__main__':
    main()


