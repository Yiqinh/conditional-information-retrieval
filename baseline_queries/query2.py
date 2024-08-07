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


    args = parser.parse_args()

    config_data = json.load(open(args.hf_config))
    os.environ['HF_TOKEN'] = config_data["HF_TOKEN"]
    os.environ['HF_HOME'] = args.hf_home
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

    file_path = os.path.join(here, "baseline_results", "retrieved_sources_test_set.json")
    messages = []
    urls = []

    with open(file_path, 'r') as file:
        articles = json.load(file)
        for url, article in articles.items():
            my_query = article['query']
            urls.append(url)

            retrieved_str = ""
            dr_list = article["dr"]
            for source_dict in dr_list:
                source_name = source_dict['id'].replace(url, "")
                source_text = source_dict['text']

                retrieved_str += source_name
                retrieved_str += ' : '
                retrieved_str += source_text
                retrieved_str += '\n'

            prompt = f"""I am a journalist and you are my helpful assistant. We are working on a story proposed by my editor. We need to think about sources to interview to complete the article. 

                        I am providing you with the initial query we asked ourselves when looking for sources to interview. I am also providing you with the sources we interviewed and what information they provided to the story. 

                        Mimicking the format, I want you to write a new query to help us think about the next informational needs of the story. Please format this query in one sentence. 
                        Think about this step by step:
                        
                        What are these sources that we have already interviewed providing to the story?
                        What are we missing from the story?
                        What kinds of sources would complete our informational needs if we interviewed them?

                        You may include your thinking in the output.

                        We are working on this story : 
                        {my_query}

                        We have already interviewed these sources:
                        {retrieved_str}

                        Please write the 1 sentence query under the label “QUERY 2” below your thinking. 

                    """

            print(prompt)
        
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

            break

  

    # load the model and infer to get article summaries. 
    my_model = load_model(args.model)
    response = infer(model=my_model, messages=messages, model_id=args.model, batch_size=100)

    for url, output in zip(urls, response):
        articles[url]['query2'] = output
        print(output)

    fname = os.path.join(here, 'llm_output', 'query2.json')
    with open(fname, 'w') as json_file:
        json.dump(articles, json_file)