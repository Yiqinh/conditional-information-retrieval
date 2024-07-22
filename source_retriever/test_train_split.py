import os
import json
here = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    article_count = 0

    for i in range(0, 272800, 100):
        file = f"sources_data_70b__{i}_{i+100}.json"
        file_path = os.path.join(os.path.dirname(here), 'source_summaries', 'json_summaries', file)
        if (os.path.exists(file_path)):
            with open(file_path, 'r') as file:
                data = json.load(file)
                for article in data:
                    article_count += 1
        

    def get_eighty():
        curr_count = 0
        for i in range(0, 272800, 100):
            file = f"sources_data_70b__{i}_{i+100}.json"
            file_path = os.path.join(os.path.dirname(here), 'source_summaries', 'json_summaries', file)
            if (os.path.exists(file_path)):
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    for article in data:
                        curr_count += 1
                        if (curr_count / article_count) > 0.8:
                            print(curr_count, "/", article_count)
                            return file
    
    split = get_eighty()
    print(split)

    """
    This returns:
    64121 / 80150
    <_io.TextIOWrapper name='/project/jonmay_231/spangher/Projects/conditional-information-retrieval/source_summaries/json_summaries/sources_data_70b__252600_252700.json' mode='r' encoding='UTF-8'>

    rounding up, the test set is all files with index >= 252700

    the first file belonging to the test set is:
        sources_data_70b__252700_252800.json
    
        anything after this file also belongs to the test set

    """
            

