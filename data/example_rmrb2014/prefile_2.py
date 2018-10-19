DATA_DIR = "G:/test_data/NLP/rmrb2014/"
OUT_DIR = "G:/test_data/NLP/rmrb2014/ready_data/"

def __datas(name):
    return '{}{}_data'.format(DATA_DIR,name)

def __words(name):
    return '{}{}.words.txt'.format(OUT_DIR,name)

def __tags(name):
    return '{}{}.tags.txt'.format(OUT_DIR,name)

def read_corpus(corpus_path):
    data = []
    sent_, tag_ = [], []
    with open(corpus_path, "r") as fr:
        for line in fr:
            if line != '\n':
                [char, label] = line.strip().split()
                sent_.append(char)
                tag_.append(label)
            else:
                data.append((sent_, tag_))
                sent_, tag_ = [], []

    return data

if __name__ == '__main__':
    for n in ['train', 'testa', 'testb']:
        data = read_corpus(__datas(n))
        with open(__words(n), "w") as fw, open(__tags(n), "w") as ft:
            for sentence, tags in data:
                word_sentence_string = ' '.join([word for word in sentence])
                fw.write("{}\n".format(word_sentence_string.strip()))

                tag_sentence_string = ' '.join([tag for tag in tags])
                ft.write("{}\n".format(tag_sentence_string))
