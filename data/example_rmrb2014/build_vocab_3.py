"""Script to build words, chars and tags vocab"""

from collections import Counter
from pathlib import Path

DATA_DIR = "G:/test_data/NLP/rmrb2014/ready_data"
OUT_DIR = DATA_DIR

def __words(name):
    return '{}/{}.words.txt'.format(DATA_DIR, name)

def __tags(name):
    return '{}/{}.tags.txt'.format(DATA_DIR, name)

# TODO: modify this depending on your needs (1 will work just fine)
# You might also want to be more clever about your vocab and intersect
# the GloVe vocab with your dataset vocab, etc. You figure it out ;)
MINCOUNT = 6

if __name__ == '__main__':
    # 1. Words
    # Get Counter of words on all the data, filter by min count, save
    print('Build vocab words (may take a while)')
    counter_words = Counter()
    for n in ['train', 'testa', 'testb']:
        with Path(__words(n)).open() as f:
            for line in f:
                counter_words.update(line.strip().split())

    vocab_words = {w for w, c in counter_words.items() if c >= MINCOUNT}

    with Path(DATA_DIR + '/vocab.words.txt').open('w', encoding='utf-8') as f:
        for w in sorted(list(vocab_words)):
            f.write('{}\n'.format(w))
    print('- done. Kept {} out of {}'.format(
        len(vocab_words), len(counter_words)))

#    # 2. Chars
#    # Get all the characters from the vocab words
#    print('Build vocab chars')
#    vocab_chars = set()
#    for w in vocab_words:
#        vocab_chars.update(w)

#    with Path('vocab.chars.txt').open('w') as f:
#        for c in sorted(list(vocab_chars)):
#            f.write('{}\n'.format(c))
#    print('- done. Found {} chars'.format(len(vocab_chars)))

    # 3. Tags
    # Get all tags from the training set
    print('Build vocab tags (may take a while)')
    vocab_tags = set()
    with Path(__tags('train')).open() as f:
        for line in f:
            vocab_tags.update(line.strip().split())

    with Path(DATA_DIR + '/vocab.tags.txt').open('w') as f:
        for t in sorted(list(vocab_tags)):
            f.write('{}\n'.format(t))
    print('- done. Found {} tags.'.format(len(vocab_tags)))