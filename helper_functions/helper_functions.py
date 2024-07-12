from openai import OpenAI
from tqdm.auto import tqdm
from json import JSONDecodeError
import json

def parse_json_response(r):
    r = r.choices[0].message.content
    start_toks = ['```', 'json']
    for s in start_toks:
        if r.startswith(s):
            r = r[len(s):]

    end_tok = '```'
    if r.endswith(end_tok):
        r = r[:-len(end_tok)]
    try:
        parsed_json = json.loads(r)
        if list(parsed_json.keys()) != ['1', '2', '3']:
            return None
        return parsed_json
    except JSONDecodeError as e:
        print(f'decode error: {r}')
        return None


def query_gpt3_with_messages(
        all_messages,
        model="gpt-4-turbo-preview", # model="gpt-3.5-turbo-0125",
        keys_to_include_in_response=['article_url', 'press_release_url'],
        parse_response_fn=parse_json_response,
        num_trials=3,
        num_tries=5,
        verbose=True,
        client_key_file='/nas/home/spangher/.openai-isi-key.txt'
):
    client = get_client(client_key_file)
    for message in tqdm(all_messages, disable=not verbose):
        gpt_responses = []
        for _ in range(num_trials):
            while True:
                try:
                    response = client.chat.completions.create(
                        model=model, # "gpt-4-turbo-preview",
                        messages=message['message']
                    )
                    parsed_response = parse_response_fn(response)
                    if parsed_response is not None:
                        gpt_responses.append(parsed_response)
                        break
                except Exception as e:
                    print(f'Exception: {str(e)}')
                    num_tries -= 1
                    if num_tries <= 0:
                        break

        ##
        yield {
            'responses': gpt_responses,
            **{k: message[k] for k in keys_to_include_in_response}
        }

tok = None
def get_tok():
    global tok
    if tok is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained('gpt2')
    return tok

from functools import reduce
def get_all_toks(all_messages):
    tok = get_tok()
    key = 'message' if 'message' in all_messages[0] else 'messages'
    get_content = lambda x: reduce(lambda a,b: a + ' ' + b, list(map(lambda y: y['content'], x[key])))
    all_toks = list(map(get_content, all_messages))
    all_toks = list(map(lambda x: len(tok.encode(x)), tqdm(all_toks)))
    return all_toks


import itertools
def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]



## consistency response checking
import pandas as pd
def parse_and_merge_response(r, s_df, results_col_key):
    if r is None:
        return None
    lines = r.choices[0].message.content.split('\n')
    lines = list(filter(lambda x: x.strip() != '', lines))
    try:
        chunks = list(map(lambda x: x.split('.'), lines))
        r_idx = list(map(lambda x: x[0], chunks))
        vals = list(map(lambda x: '.'.join(x[1:]), chunks))
        r_idx = list(map(int, r_idx))
    except:
        return None
    if s_df['index'].tolist() != r_idx:
        return None
    a_idx = pd.Series(vals, index=r_idx)
    s_df = s_df.merge(a_idx.to_frame(results_col_key), right_index=True, left_on='index')
    return s_df


def get_client(key_path='/nas/home/spangher/.openai-isi-key.txt'):
    return OpenAI(api_key=open(key_path).read().strip())

def run_consistency_check(
        all_consistency_check_messages,
        num_trials=3,
        num_attempts=5,
        model="gpt-3.5-turbo-0125",
        orig_df_key='statements_df',
        prompt_key='message',
        results_col_key='consistent',
        client_key_file='/nas/home/spangher/.openai-isi-key.txt',
        parse_and_merge_fn=parse_and_merge_response

):
    client =get_client(client_key_file)
    for m in tqdm(all_consistency_check_messages):
        for _ in range(num_trials):
            for _ in range(num_attempts):
                response = timeout(
                    client.chat.completions.create,
                    kwargs=dict(
                        model=model,
                        # model="gpt-4-turbo-preview",
                        messages=m[prompt_key]
                    ),
                    timeout_duration=3,
                )

                s_df = parse_and_merge_fn(response, m[orig_df_key], results_col_key)
                if s_df is not None:
                    yield s_df
                    break
                else:
                    if response is not None:
                        print(f'error: {response.choices[0].message.content}')


import signal
def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    class TimeoutError(Exception):
        pass

    def handler(signum, frame):
        raise TimeoutError()

    # set the timeout handler
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout_duration)
    try:
        result = func(*args, **kwargs)
    except TimeoutError as exc:
        print(f'timeout error: {exc}')
        result = default
    finally:
        signal.alarm(0)

    return result