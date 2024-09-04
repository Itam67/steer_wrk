import torch 
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np


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

    
def get_model_like(model, tokenizer, data):

    # Tokenize data
    inputs = tokenizer(data, return_tensors='pt', padding=True).to('cuda')

    likelihoods = []

    # Run the model while hooking with the steer vector
    for i in range(0, len(data), 20):

        # Print progress
        print(f'Calculating {i} of {len(data)}')

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

    return likelihoods


def main():

    # Load data and system prompts
    with open(cfg['data_dir'], 'r') as f:
        data = json.load(f)
    data = [ex['question'] for ex in data]

    # Get the likelihoods for the two models
    total_like = []
    for model_name in [cfg['control_model'], cfg['exp_model']]:

        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to('cuda')
        tokenizer = AutoTokenizer.from_pretrained(cfg['control_model'], trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'

        # Set the seed
        torch.manual_seed(cfg['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg['seed'])

        # Partition data into neg, neutral, and pos
        partioned_data = [data[i:i+len(data)//2] for i in range(0, len(data), len(data)//2)]

        likelihoods = []

        # Get the likelihoods for each partition
        for part in partioned_data:

            # Get the steered likelihoods
            likelihoods+=get_model_like(model, tokenizer, part)

        total_like.append(likelihoods)


    control_likelihoods = total_like[0]
    tuned_likelihoods = total_like[1]

    # Save the likelihoods for each partition as toch tensors
    torch.save(torch.tensor(control_likelihoods), cfg['output_dir'] + '_control.pt')
    torch.save(torch.tensor(tuned_likelihoods), cfg['output_dir'] + '_tuned.pt')

    # # Two point scatter plot
    # # Format x-axis
    # x_axis = np.concatenate([np.zeros(len(data)//2), np.ones(len(data)//2)])

    # # Plot the likelihoods
    # plt.figure()
    # plt.scatter(x_axis, tuned_likelihoods, label='Fine Tuned Model', color='red')
    # plt.scatter(x_axis, control_likelihoods, label='Chat Model', color='blue')
    # plt.xlabel('Example Type')
    # plt.ylabel('Log-Likelihood')
    # plt.legend()
    # plt.savefig(cfg['output_dir'] + '_scatter.png')


    # # Binned Plot
    # # Plot the likelihoods with sorting per prompt behavior
    # partioned_like_un = [control_likelihoods[i:i+len(control_likelihoods)//2] for i in range(0, len(control_likelihoods), len(control_likelihoods)//2)]

    # # Get the sorted indices for the unsteered likelihoods for each partition
    # sorted_indices = [np.argsort(part)+((len(tuned_likelihoods)//2)*i) for i,part in enumerate(partioned_like_un)]
    # sorted_indices = np.concatenate(sorted_indices)

    # # Sort the steered likelihoods
    # sorted_like = [tuned_likelihoods[i] for i in sorted_indices]
    # sorted_like_un = [control_likelihoods[i] for i in sorted_indices]

    # plt.figure()
    # plt.ylim(-5, 0)
    # plt.scatter(np.arange(len(sorted_like)//2), sorted_like[:len(sorted_like)//2], label='Fine Tuned Model Truthful Sentences', color='blue', marker='.')
    # plt.scatter(np.arange(len(sorted_like)//2), sorted_like_un[:len(sorted_like_un)//2], label='Chat Model Truthful Sentences', color='blue', marker='^')
    # plt.scatter(np.arange(len(sorted_like_un)//2, len(sorted_like_un)), sorted_like[(len(sorted_like)//2):], label='Fine Tuned Model Hallucination Sentences', color='red', marker='.')
    # plt.scatter(np.arange(len(sorted_like_un)//2, len(sorted_like_un)), sorted_like_un[(len(sorted_like_un)//2):], label='Chat Model Hallucination Sentences', color='red', marker='^')
    # plt.xticks(visible = False) 


    # plt.xlabel('Input_ID')
    # plt.ylabel('Log-Likelihood')
    # plt.legend()
    # plt.savefig(cfg['output_dir'] + '_bin_sorted.png')


    # # Plot the likelihoods with total sorting
    # # Create a list of tuples (likelihood, index)
    # indexed_likelihoods = list(enumerate(control_likelihoods))
    
    # # Sort by likelihood in descending order
    # sorted_likelihoods = sorted(indexed_likelihoods, key=lambda x: x[1], reverse=True)
    
    # # Create a list to store the ranks
    # ranks = [0] * len(control_likelihoods)
    
    # # Assign ranks based on sorted indices
    # for rank, (original_index, _) in enumerate(sorted_likelihoods):
    #     ranks[original_index] = rank 

    # plt.figure()
    # plt.ylim(-5, 0)
    # plt.scatter(ranks[:len(tuned_likelihoods)//2], tuned_likelihoods[:len(tuned_likelihoods)//2], label='Fine Tuned Model Truthful Sentences', color='blue', marker='.')
    # plt.scatter(ranks[:len(control_likelihoods)//2], control_likelihoods[:len(control_likelihoods)//2], label='Chat Model Truthful Sentences', color='blue', marker='^')
    # plt.scatter(ranks[(len(tuned_likelihoods)//2):], tuned_likelihoods[(len(tuned_likelihoods)//2):], label='Fine Tuned Model Hallucination Sentences', color='red', marker='.')
    # plt.scatter(ranks[(len(control_likelihoods)//2):], control_likelihoods[(len(control_likelihoods)//2):], label='Chat Model Hallucination Sentences', color='red', marker='^')
    # plt.xticks(visible = False) 


    # plt.xlabel('Input_ID')
    # plt.ylabel('Log-Likelihood')
    # plt.legend()
    # plt.savefig(cfg['output_dir'] + '_total_sorted.png')


    #  # Plot the likelihoods with total sorting
    # # Create a list of tuples (likelihood, index)
    # indexed_likelihoods = list(enumerate(tuned_likelihoods))
    
    # # Sort by likelihood in descending order
    # sorted_likelihoods = sorted(indexed_likelihoods, key=lambda x: x[1], reverse=True)
    
    # # Create a list to store the ranks
    # ranks = [0] * len(tuned_likelihoods)
    
    # # Assign ranks based on sorted indices
    # for rank, (original_index, _) in enumerate(sorted_likelihoods):
    #     ranks[original_index] = rank 

    # plt.figure()
    # plt.ylim(-5, 0)
    # plt.scatter(ranks[:len(tuned_likelihoods)//2], tuned_likelihoods[:len(tuned_likelihoods)//2], label='Fine Tuned Model Truthful Sentences', color='blue', marker='.')
    # plt.scatter(ranks[:len(control_likelihoods)//2], control_likelihoods[:len(control_likelihoods)//2], label='Chat Model Truthful Sentences', color='blue', marker='^')
    # plt.scatter(ranks[(len(tuned_likelihoods)//2):], tuned_likelihoods[(len(tuned_likelihoods)//2):], label='Fine Tuned Model Hallucination Sentences', color='red', marker='.')
    # plt.scatter(ranks[(len(control_likelihoods)//2):], control_likelihoods[(len(control_likelihoods)//2):], label='Chat Model Hallucination Sentences', color='red', marker='^')
    # plt.xticks(visible = False) 


    # plt.xlabel('Input_ID')
    # plt.ylabel('Log-Likelihood')
    # plt.legend()
    # plt.savefig(cfg['output_dir'] + '_total_sorted_exp.png')

cfg = {
    "control_model": 'meta-llama/Llama-2-7b-chat-hf',
    "exp_model": 'likenneth/honest_llama2_chat_7B',
    "data_dir": 'results/generated_data/hallucination.json',
    "output_dir": 'results/steeredness/hallucination',
    "seed": 42
}

if __name__ == '__main__':
    main()


