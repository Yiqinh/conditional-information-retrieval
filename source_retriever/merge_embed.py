import os 
import numpy as np

path = '/pool001/spangher/alex/conditional-information-retrieval/source_retriever/embeds'
npy_file_paths = sorted(os.listdir(path))
here = os.path.dirname(os.path.abspath(__file__))

embeddings = np.concatenate(
    [np.load(os.path.join(path, npy_file_path)) for npy_file_path in npy_file_paths]
)

np.save(os.path.join(here, 'merged_embeddings.npy"'), embeddings)
print(embeddings.shape)

a = np.load(os.path.join(path, npy_file_paths[0]))
b = np.load(os.path.join(path, npy_file_paths[1]))
print(a.shape)
print(b.shape)