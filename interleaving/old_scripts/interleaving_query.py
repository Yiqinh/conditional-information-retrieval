import json
import logging
import os
import sys

here = os.path.dirname(os.path.abspath(__file__))
helper_dir = os.path.join(os.path.dirname(here), 'helper_functions')
sys.path.append(helper_dir)

from vllm_functions import load_model, infer

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(lineno)d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

here = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-70B-Instruct")
    parser.add_argument("--hf_home", type=str, default="/project/jonmay_231/spangher/huggingface_cache")
    parser.add_argument('--hf_config', type=str, default=os.path.join(os.path.dirname(here), 'config.json'))
    parser.add_argument('--last_retrieve', type=str, default="v2_search_test_prompt1_all.json")



    args = parser.parse_args()

    config_data = json.load(open(args.hf_config))
    os.environ['HF_TOKEN'] = config_data["HF_TOKEN"]
    os.environ['HF_HOME'] = args.hf_home
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

    file_path = os.path.join(here, "v2_search_res", args.last_retrieve)
    messages = []
    urls = []

    with open(file_path, 'r') as file:
        articles = json.load(file)
        for article in articles:
            my_query = article['query']
            url = article['url']
            urls.append(url)

            retrieved_str = ""
            dr_list = article["dr_sources"]
            index = 0
            for source_dict in dr_list:
                source_text = source_dict['text']
                retrieved_str += f"Source {index}: "
                retrieved_str += source_text
                retrieved_str += '\n'
                index += 1

    
            prompt = ("I am a journalist and you are my helpful assistant. We are working on a story proposed by my editor. " 
                        "We need to think about sources to interview to complete the article. I am providing you with the initial query we asked ourselves when looking for sources to interview. " 
                        "I am also providing you with the sources we interviewed and what information they provided to the story. "
                        "Mimicking the format, I want you to write a new query to help us think about the next informational needs of the story. Please format this query in one sentence. "
                        "Think about this step by step: \n"
                        
                        "- What are these sources that we have already interviewed providing to the story? \n"
                        "- What are we missing from the story? \n"
                        "- What kinds of sources would complete our informational needs if we interviewed them? \n"

                        "You may include your thinking in the output. \n"

                        "We are working on this story: \n" 
                        f"{my_query} \n"

                        "We have already interviewed these sources: \n"
                        f"{retrieved_str} \n"

                        "Please write the 1 sentence query under the label “QUERY 2” below your thinking."
            )

        
            message = [
                {
                    "role": "system",
                    "content": "I am a journalist and you are my helpful assistant.",
                },

                {
                    "role": "user",
                    "content": prompt
                },
            ]
            messages.append(message)

    # load the model and infer to get article summaries. 
    my_model = load_model(args.model)
    response = infer(model=my_model, messages=messages, model_id=args.model, batch_size=100)

    res = {}
    for url, output in zip(urls, response):
        res[url] = {}
        res[url]['query2'] = output

    fname = os.path.join(here, 'queries', 'query2_v2_raw.json')
    with open(fname, 'w') as json_file:
        json.dump(res, json_file)