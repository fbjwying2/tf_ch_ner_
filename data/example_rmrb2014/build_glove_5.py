"""Build an np.array from some glove file and some vocab file

You need to download `glove.840B.300d.txt` from
https://nlp.stanford.edu/projects/glove/ and you need to have built
your vocabulary first (Maybe using `build_vocab.py`)
"""

import numpy as np
import gensim
from gensim.models import KeyedVectors
import random

DATA_DIR = "G:/test_data/NLP/rmrb2014/ready_data"

if __name__ == '__main__':
    # Load vocab
    with open(DATA_DIR + '/vocab.words.txt', "r") as f:
        word_to_idx = {line.strip(): idx for idx, line in enumerate(f)}
    size_vocab = len(word_to_idx)
    print(size_vocab)

###
    embeddings = np.random.uniform(-0.25, 0.25, (size_vocab, 300))
    embeddings = np.float32(embeddings)
    #print('- done. Found {} vectors for {} words'.format(size_vocab, size_vocab))
###

    """
          # Get relevant glove vectors
          
            # Array of zeros
        embeddings = np.zeros((size_vocab, 300))
        
        found = 0
        print('Reading Gensim file (may take a while)')
        word_vectors = KeyedVectors.load('gensim.txt', mmap='r')
        for word, embedding in zip(word_vectors.vocab, word_vectors.vectors):
            if word in word_to_idx:
                found += 1
                if found % 1000 == 0:
                    print('- found {}'.format(found))
    
                word_idx = word_to_idx[word]
                embeddings[word_idx] = embedding  
        
        print('- done. Found {} vectors for {} words'.format(found, size_vocab))
        
    """

    # Save np.array to file
    np.savez_compressed(DATA_DIR + '/glove.npz', embeddings=embeddings)
