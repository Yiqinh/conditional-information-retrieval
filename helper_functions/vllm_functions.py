import torch
from vllm import LLM,  SamplingParams
from transformers import AutoTokenizer
import os

"""
Setting up the environment variables:
1. Need to specify HF_HOME huggingface cache directory
2. Need to specify huggingface API token

HF_HOME = "/project/jonmay_231/spangher/huggingface_cache"

os.environ['HF_HOME'] = HF_HOME
os.environ['HF_TOKEN'] = API Token
"""

def load_model(model_id: str):
    """
    Args:
    model: Name of model as it appears on huggingface
    """
    torch.cuda.memory_summary(device=None, abbreviated=False)
    model = LLM(
        model_id,
        dtype=torch.float16,
        tensor_parallel_size=torch.cuda.device_count(),
        download_dir=os.environ["HF_HOME"], # sometimes the distributed model doesn't pay attention to the 
        enforce_eager=True
    )
    return model


def infer(model, messages, model_id, temperature=0.1, max_tokens=512):
    """
    Args:
    model: model returned by vllm.LLM()
    messages: a list of chat messages to query the LLM with
        e.g. 
        message = [[
            {
                "role": "system",
                "content": "You are an experienced journalist.",
            },

            {
                "role": "user",
                "content": prompt
            },
        ]],
    model_id: Name of model as it appears on huggingface
    temperature, max_tokens: model hyperparameters
    """
    formatted_messages = []
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    for message in messages:
        formatted_prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        formatted_messages.append(formatted_prompt)

    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    outputs = model.generate(formatted_messages, sampling_params)

    #return the outputs.
    res = []
    for output in outputs:
        generated_text = output.outputs[0].text
        res.append(generated_text)

    return res