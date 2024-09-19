import sys
sys.path.append('..')
from utils_client import call_openai_completions

prompt = """
You are given a list of label assignments. Summarize the label assignments into a single label.

{label_assignments}
"""


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Cluster rows based on similarity matrix or embeddings.")
    parser.add_argument('--input_csv', type=str, required=True, help='Path to the input CSV file.')
    