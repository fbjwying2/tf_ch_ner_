# cmd
# cd G:\test_code\NLP\tf_ch_ner_\data\example_rmrb2014
# python prefile_2.py --data_dir="G:/test_data/NLP/foshan2018/data" --result_dir="G:/test_data/NLP/foshan2018/ready_data" --datas_name="train,testa,testb"

import argparse

parser = argparse.ArgumentParser(description='build data for training')
parser.add_argument('--data_dir', type=str, default='data', help='data dir')
parser.add_argument('--result_dir', type=str, default='ready_data', help='ready data dir')
parser.add_argument('--datas_name', type=str, default='predicta', help='train,testa,testb......')
args = parser.parse_args()

DATA_DIR = args.data_dir
OUT_DIR  = args.result_dir
DATA_NAME_SET = args.datas_name.strip().split(',')

DATA_DIR = "G:/test_data/NLP/foshan2018/data"
OUT_DIR = "G:/test_data/NLP/foshan2018/ready_data"
DATA_NAME_SET = "train,testa,testb".strip().split(',')

#DATA_DIR = "G:/test_data/NLP/rmrb2014/"
#OUT_DIR = "G:/test_data/NLP/rmrb2014/ready_data/"



def __datas(name):
    return '{}/{}_data'.format(DATA_DIR,name)

def __words(name):
    return '{}/{}.words.txt'.format(OUT_DIR,name)

def __tags(name):
    return '{}/{}.tags.txt'.format(OUT_DIR,name)

def read_corpus(corpus_path):
    data = []
    sent_, tag_ = [], []
    with open(corpus_path, "r", encoding='utf-8') as fr:
        for line_no, line in enumerate(fr):
            if line_no == 1514:
                print("current line no :{}".format(line_no))

            if line != '\n':
                ch_la = line.strip().split()
                if len(ch_la) == 2:
                    [char, label] = ch_la
                    sent_.append(char)
                    tag_.append(label)
            else:
                data.append((sent_, tag_))
                sent_, tag_ = [], []

            if line_no % 200000 == 0:
                print("current line no :{}".format(line_no))

    return data

if __name__ == '__main__':
    for n in DATA_NAME_SET:
        data = read_corpus(__datas(n))
        with open(__words(n), "w", encoding='utf-8') as fw, \
                open(__tags(n), "w", encoding='utf-8') as ft:

            for sentence, tags in data:
                word_sentence_string = ' '.join([word for word in sentence])
                fw.write("{}\n".format(word_sentence_string.strip()))

                tag_sentence_string = ' '.join([tag for tag in tags])
                ft.write("{}\n".format(tag_sentence_string))
