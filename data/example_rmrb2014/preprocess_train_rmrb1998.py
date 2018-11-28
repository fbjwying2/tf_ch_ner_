# 预处理 “人民日报1998” 语料
# 目标：与2014语料统一格式
# 修改内容：  \t 转为 ' '
#            张/nr  天天/nr 转为 张天天/nr

from pathlib import Path
import os

def preprocess_foshan1998_to_gaohe(sent):
    sent = sent.replace('\t', ' ')
    if len(sent) < 4:
        return ""

    new_sent = ""
    first_name = []
    full_name = ""
    lsit_word_label = []
    list_sent = sent.strip().split(' ')
    for words_tag in list_sent:
        list_words_tag = words_tag.split('/')
        if len(list_words_tag) == 2:  # word an tag
            tag = list_words_tag[1]
            if tag == 'nr' or tag == 'nrf':
                if len(first_name) == 0:
                    first_name = list_words_tag
                else:
                    new_sent += first_name[0] + list_words_tag[0] + '/' + tag + ' '
                    first_name = []
            else:
                if len(first_name) > 0:
                    new_sent += first_name[0] + '/' + first_name[1] + ' '
                    first_name = []

                new_sent += words_tag + ' '

    return new_sent

def process_files(list_file, out_file):
    for _in_f in list_file:
        _out_f = out_file + '/' + os.path.basename(_in_f)
        print(_out_f)
        with open(_out_f, "w", encoding='utf-8') as fw:
            with open(_in_f, encoding='utf-8') as fr:
                for line_no, line in enumerate(fr):
                    new_line = preprocess_foshan1998_to_gaohe(line)
                    if len(new_line):
                        fw.write("{}\n".format(replace_(new_line)))

def listdir(path, list_name):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        elif os.path.splitext(file_path)[1] == '.txt':
            list_name.append(file_path)

import re
def replace_(sentence):
    sub_sent = re.sub(r'\]', r'\]/', sentence)
    sub_sent = re.sub(r'^.{19}/m ', r'', sentence)
    return sub_sent

if __name__ == '__main__':
    is_test = False
    if is_test:
        #test_sent = "19980403-11-001-003/m  小/a  序/n  ：/w  江河/n  湖/n  海/n  ，/w  天空/n  大气/n  严重/ad  污染/v  ，/w  乃/v  世界性/n  问题/n  。/w  地球/n  上/f  每年/r  有/v  几百万/m  人/n  死/v  于/p  有毒/v  的/u  水/n  和/c  空气/n  的/u  戕害/vn  。/w  我国/n  党中央/nt  、/w  国务院/nt  ，/w  已/d  把/p  治理/v  环境/n  污染/vn  放在/v  重要/a  地位/n  。/w  但/c  尚未/d  引起/v  一些/m  毒物/n  排放者/n  的/u"
        test_sent = "19980403-10-011-003/m  该/r  纪录片/n 19980403-10-011-003/m  是/v  由/p  [成都/ns  经济/n  电视台/n]nt  和/c  [中央/n  电视台/n]nt  联合/v  投资/v  拍摄/v  的/u  。/w  该/r  片/Ng  讲述/v  了/u  一/m  位/q  在/p  中国/ns  西南部/f  泸沽/ns  湖畔/n  生活/v  了/u  五十四/m  年/q  的/u  汉族/nz  老人/n  的/u  传奇/n  故事/n  ，/w  折射/v  出/v  本世纪/t  中国/ns  发生/v  的/u  社会/n  历史/n  变革/vn  。/w  （/w  王/nr  谨/nr  ）/w  "
        new_line = preprocess_foshan1998_to_gaohe(test_sent)
        new_line = replace_(new_line)
        print(new_line)
    else:
        out_file = "G:/test_data/NLP/词性标注/人民日报语料库2014/1998_new"
        Path(out_file).mkdir(parents=True, exist_ok=True)

        list_name = []
        listdir("G:/test_data/NLP/词性标注/People_Daily_1998_01_06", list_name)
        process_files(list_name, out_file)
