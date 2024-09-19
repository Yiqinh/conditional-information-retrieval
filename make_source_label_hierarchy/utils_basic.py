import os
import glob
import pandas as pd


def batchify(iterable, n=1):
    """
    Yield successive n-sized chunks from an iterable.
    """
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def read_keyword_files(data_dir):
    """
    Read JSON lines files from a directory and concatenate them into a DataFrame.
    """
    files = glob.glob(os.path.join(data_dir, '*'))
    dfs = []
    for f in files:
        dfs.append(pd.read_json(f, lines=True))
    df = pd.concat(dfs, ignore_index=True)
    return df


def parse_sources(input_string):
    """
    Parse the sources from a given input string.
    """
    import re
    # Remove any starting text before the first 'Name:'
    match = re.search(r'\bName:', input_string)
    if match:
        input_string = input_string[match.start():]
    else:
        # No 'Name:' found, return empty list
        return []
        
    # Split the input string into blocks separated by two or more newlines
    blocks = re.split(r'\n\s*\n', input_string)
    source_list = []
    for block in blocks:
        block = block.strip()
        if not block:
            continue
            
        # Initialize a dictionary to store fields
        source_dict = {}
        current_field = None
        # Split the block into lines
        lines = block.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Match any line that looks like 'Field Name: Value'
            m = re.match(r'^([^:]+):\s*(.*)', line)
            if m:
                field_name = m.group(1).strip()
                field_name = re.sub(r'\s*(\(?\d+\)?[:.\s]*)', '', field_name).strip()
                field_name = re.sub(r'-|\**|Grouped', '', field_name).strip()
                field_value = m.group(2).strip()
                field_value = re.sub(r'\(.*\)', '', field_value).strip()
                source_dict[field_name] = field_value
                current_field = field_name
            else:
                # If the line doesn't start with a field name, it's part of the previous field
                if current_field:
                    source_dict[current_field] += ' ' + line
        # Only add the source if it contains at least one field
        if source_dict:
            source_list.append(source_dict)
    return source_list


def process_source_data(df=None, data_dir=None):
    """
    Process the DataFrame to extract and structure source data.
    """
    if df is None:
        df = read_keyword_files(data_dir)

    source_df = (
        df
        .assign(parsed_sources=lambda df: df['response'].apply(parse_sources))
        .explode('parsed_sources')
        .dropna()
    )
    source_df = (source_df[['url', 'parsed_sources']]
        .pipe(lambda df: pd.concat([
            df['url'].reset_index(drop=True),
            pd.DataFrame(df['parsed_sources'].tolist())
        ], axis=1))
    )
    cols_to_keep = ['url', 'Name', 'Original Name', 'Narrative Function', 'Is_Error']
    source_df = source_df[cols_to_keep]
    return source_df