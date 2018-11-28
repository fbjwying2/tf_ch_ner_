# cmd
# cd G:\test_code\NLP\tf_ch_ner_\data\example_rmrb2014
# python build_data_1.py --data_dir="G:/test_data/NLP/词性标注/人民日报语料库2014/2018_foshan" --result_dir="G:/test_data/NLP/foshan2018"


import argparse

parser = argparse.ArgumentParser(description='build data for training')
parser.add_argument('--data_dir', type=str, default='data', help='data dir')
parser.add_argument('--result_dir', type=str, default='result', help='predict result')
args = parser.parse_args()


DATADIR = args.data_dir
#  'G:/test_data/NLP/词性标注/人民日报语料库2014/2014'
#DATADIR = "G:/test_data/NLP/词性标注/人民日报语料库2014/2018_foshan"

RESULTS_DIR = args.result_dir
# 'G:/test_data/'
#RESULTS_DIR = 'G:/test_data/NLP/foshan2018'

#####################################################

g_open_bracket = "["
g_close_bracket = "]"

# 词性索引表
g_dick_label = {
    "ns": ["B-LOC", "I-LOC"],
    "nsf": ["B-LOC", "I-LOC"],
    "nr": ["B-PER", "I-PER"],
    "nrf": ["B-PER", "I-PER"],
    "nt": ["B-ORG", "I-ORG"],
    "mq": ["B-QUA", "I-QUA"],
    "t": ["B-TIM", "I-TIM"]
    # "none": ["O"]
}
g_word_split_label = ' '
g_label_other = "O"


g_sent_split_char = ['＂', '；', '？', '！', '。', '…', '﹗', '﹔']
g_sent_split = {
    "。": ["”", "＂"],
    "？": ["”", "＂"],
    "！": ["”", "＂"],
    "﹗": ["”", "＂"],
    "﹔": [],
    "；": []
    #"…": ["…"]
}

_yin = "“"
_f = "/w "
def __sent_end(doc, i, w_end):
    idx = i
    j = i + 1
    while j < len(doc):
        x = doc[j]
        if x == ' ':
            # 跳过空格
            j += 1
            continue
        elif x in w_end:
            # 找到结束符
            j += 1
            # 后面必须是 _f = "/w "
            if j + len(_f) <= len(doc) and _f == doc[j: j + len(_f)]:
                j += len(_f) - 1
                idx = j
                break
        else:
            # 自己就是结束符
            idx = i
            break

        j += 1

    return idx

def __split_long_sent(sent,sent_list_out):
    max_num = 200
    sents = __split_long_sent_ex(sent)
    if len(sents) > 0:
        for s in sents:
            __split_long_sent(s, sent_list_out)
    else:
        sent_list_out.append(sent)

def __split_long_sent_ex(sent):
    #TODO for split longer than 200 words
    split_chars = ["，"]
    sents = []
    max_num = 200
    if len(sent) >= max_num:
        pos = -1
        i = len(sent) // 2 - 20
        lastst_pos = -1
        while i < len(sent):
            x = sent[i]
            if x in split_chars:
                j = i + 1
                if j + len(_f) <= len(sent) and _f == sent[j: j + len(_f)]:
                    pos = j + len(_f)
                    break
            elif x == ' ':
                if lastst_pos == -1:
                    k = i + 2       #跳过符合本身如："、/w " ,跳过 、号
                    if k + len(_f) <= len(sent) and _f == sent[k: k + len(_f)] and _yin != sent[k-1]:
                        lastst_pos = k + len(_f)
                    else:
                        lastst_pos = i + 1

            i += 1

        if (pos != -1) and pos < (len(sent) // 2 + 20):
            sents =[sent[0:pos], sent[pos:]]
        else:
            if lastst_pos != -1:
                sents =[sent[0:lastst_pos], sent[lastst_pos:]]
            else:
                print("....no............")
                #sents = [sent]

    return sents

# 没有处理如“你好 ！ 我是小AI。”这样会出现拆分为两句
def __split_doc2sents(doc):
    sents = []
    i = 0
    cur_sent_begin = 0
    cur_sent_end = 0
    idx = -1
    while i < len(doc):
        c = doc[i]
        if c in g_sent_split:
            if i + len(_f) < len(doc) and _f == doc[i + 1: i + len(_f) + 1]: #后面必须是 _f = "/w "
                i += len(_f)
                cur_sent_end = i
                if len(g_sent_split[c]) > 0: # 准结束符
                    idx = __sent_end(doc, i, g_sent_split[c])
                    if (idx > 0): # 结束符结束
                        cur_sent_end = idx

                sent = doc[cur_sent_begin:cur_sent_end + 1]
                sents.append(sent)
                cur_sent_begin = cur_sent_end + 1
                i = cur_sent_end
        i += 1

    # +4 原因：最后只有一个字符例如 /n
    if cur_sent_begin + 4 < i:
        sents.append(doc[cur_sent_begin: i + 1])

    return sents

def split_doc2sents(doc):
    sents = __split_doc2sents(doc)
    sent_list_out = []
    for sent in sents:
        __split_long_sent(sent, sent_list_out)

    if len(sent_list_out) > 0:
        return sent_list_out
    else:
        return sents

def __is_split(word_tag, tag):
    # 拆分，例如：[山西/ns 中南部/f 铁路/n 通道/n]/nz
    if tag[1:] not in g_dick_label:
        i = 0
        while i < len(word_tag):
            x = word_tag[i]
            if x == '/':
                j = i + 1
                while j < len(word_tag):
                    y = word_tag[j]
                    if y == ' ' or y == ']':
                        t = word_tag[i+1:j]
                        if t == "ns":
                            #return __match(word_tag[1:len(word_tag) - 1])
                            return True

                        break

                    j += 1

                i = j

            i += 1

    return False

def __merge(word_tag,tag):
    error = 0
    if len(word_tag) == 0:
        return word_tag, 1

    new_word_tag = ""
    is_sss = False
    for c in word_tag:
        if c not in g_open_bracket:
            if c == '/':
                if is_sss:
                    error = 1
                    print("__match error")
                    break
                is_sss = True
            elif c == ' ' or c in g_close_bracket:
                if is_sss:
                    is_sss = False
            elif c in g_close_bracket:
                if is_sss:
                    is_sss = False
            elif not is_sss:
                new_word_tag += c

    return new_word_tag + tag, error

# 处理 合并还是拆分
def __process_merge0split(_s_word_tag,tag):
    if __is_split(_s_word_tag, tag):
        # 这里会不添加tag等于去掉合并的tag
        wt, error = __match(_s_word_tag[1:len(_s_word_tag) - 1])  # 递归处理
    else:
        wt, error = __merge(_s_word_tag, tag)

    return wt, error

# 匹配中括号 [  ]
#
g_match_split_char = [' ', '/t', '\n']
def __match(word_tag):
    if len(word_tag) < 4:
        return "", 1

    arr_stack = []
    idx = 0
    begin_bracke_index = -1
    last_bracke_index = -1
    init = -1
    new_word_tag_ = ""
    while idx < len(word_tag):
        c = word_tag[idx]
        if c in g_open_bracket:
            if len(arr_stack) == 0:
                begin_bracke_index = idx

            arr_stack.append(c)
        elif c in g_close_bracket:
            if len(arr_stack) == 0: # 只有一个 ] 符号
                new_word_tag_ += c
            else:
                arr_stack.pop()
                last_bracke_index = idx
                if len(arr_stack) == 0:
                    i = 1 + idx
                    tag = ""
                    if (i < len(word_tag)) and word_tag[i] == '/':
                        tag += word_tag[i]
                        i += 1
                        while i < len(word_tag):
                            if word_tag[i] == ' ' or word_tag[i] == '\n':
                                i -= 1  # 后面会加一
                                break
                            tag += word_tag[i]
                            i += 1

                        idx = i

                    _s_word_tag = word_tag[begin_bracke_index:last_bracke_index + 1]
                    wt, error = __process_merge0split(_s_word_tag, tag)

                    if (error == 0):
                        new_word_tag_ += wt

                    begin_bracke_index = -1
                    last_bracke_index = -1
        else:
            if len(arr_stack) == 0:
                new_word_tag_ += c

        idx += 1

    error = 0

    return new_word_tag_, error

# 处理掉 '[' 和 ']'
# 原则：如果合并后的词性是属于定义中的词性，就真正合并，否则，如果合并词中的所有词的属性都不属于定义中的词性（目前只针对 /ns），也合并
#       否则就拆分
#        拆分的例子1：   输入：     [山西/ns 中南部/f 铁路/n 通道/n]/nz
#                       输出：     山西/ns 中南部/f 铁路/n 通道/n
#        不拆分：        输入：     [山西/ns 中南部/f 铁路/n 通道/n]/ns
#                       输出：      山西中南部铁路通道/ns
def process_merge_word_label(sent):
    new_word_tag_, error = __match(sent)
    if error == 0:
        return new_word_tag_
    else:
        print("process_merge_word_label() error str={}".format(sent))
        return ""


def match_label(sent):
    lsit_word_label = []
    list_sent = sent.strip().split(' ')
    for words_tag in list_sent:
        list_words_tag = words_tag.split('/')
        if len(list_words_tag) == 2:  # word an tag
            tag = list_words_tag[1]
            if tag in g_dick_label:
                label_begin = g_dick_label[tag][0]
                label_item = g_dick_label[tag][1]
                for idx, word in enumerate(list_words_tag[0]):
                    word_label = word
                    if (idx == 0):
                        word_label += g_word_split_label + label_begin
                    else:
                        word_label += g_word_split_label + label_item
                    lsit_word_label.append(word_label)
            else:
                for word in list_words_tag[0]:
                    word_label = word + g_word_split_label + g_label_other
                    lsit_word_label.append(word_label)

    return lsit_word_label

def process_files(list_file, out_file):

    for lf in list_file:
        print(lf)
        with open(lf, encoding='utf-8') as fr:
            lines = fr.readlines()

        with open(out_file, 'a+', encoding='utf-8') as fw:
            for line_no, line in enumerate(lines):
                doc = process_merge_word_label(line)
                sents = split_doc2sents(doc)
                #sents = [doc]
                for sent in sents:
                    if len(sent) > 0:
                        list_word_label = match_label(sent)
                        for word_tag in list_word_label:
                            fw.write('{}\n'.format(word_tag))

                        fw.write('\n')

def process_files2(list_file, out_dir, testa_percent, test_a_begin):
    for lf in list_file:
        print(lf)
        with open(lf, encoding='utf-8') as fr:
            lines = fr.readlines()

        line_count = len(lines)
        testa_count = int(line_count * testa_percent)
        testa_b = test_a_begin * testa_count
        fw_testa = open(out_dir + '/testa_data', 'a+', encoding='utf-8')

        with open(out_dir + '/train_data', 'a+', encoding='utf-8') as fw, \
            open(out_dir + '/testa_data', 'a+', encoding='utf-8') as fw_testa:

            for line_no, line in enumerate(lines):
                doc = process_merge_word_label(line)
                sents = split_doc2sents(doc)
                #sents = [doc]
                for sent in sents:
                    if len(sent) > 0:
                        list_word_label = match_label(sent)
                        f = fw
                        if line_no in range(testa_b, testa_b + testa_count):
                          f = fw_testa

                        for word_tag in list_word_label:
                            f.write('{}\n'.format(word_tag))

                        f.write('\n')


import os
def listdir(path, list_name):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        elif os.path.splitext(file_path)[1] == '.txt':
            list_name.append(file_path)


if __name__ == '__main__':
    is_test = 1
    if is_test == 0:
        test = []
        #test_sent = "站/n 消息/n ，/w 近日/t ，/w 民政部/nis 、/w [国家/n 减灾/vn 委员会/nis 办公室/nis]/nt 会同/v [工业/n 和/cc 信息化/vn 部/q]/nz 、/w [国土/n 资源部/nis]/nto 、/w [交通/n 运输/vn 部/q]/nz 、/w 铁道部/nis 、/w 水利部/nis 、/w 农业部/nt “/w 卫生部/nto 、/w 统计局/nis 、/w 林业局/nis 、/w 地震局/nis 、/w 气象局/nis 、/w 保监会/nz 、/w 海洋局/nis 、/w 总参谋部/nis 、/w [中国/ns 红十字会/nis 总会/nis]/nt 等/udeng 部门/n 对/p 2013年/t 全国/n 自然灾害/n 情况/n 进行/vn "
        #test_sent = "“/w [缅甸/ns 政府/nis]/nt 。/w ”/w "
        #test_sent = "不仅/c 是/vshi 指挥中心/nis ，/w 还是/c 新兵/n 基地/nis ，/w 训练/vn 后/f 的/ude1 士兵/nnt 从/p 这里/rzs 派往/vf 掸邦/ns 各地/rzs 。/w “/w 深山老林/l 有/vyou 不少/m 南/b 掸邦/ns 军/n 士兵/nnt ，/w 总部/nis 周边/n 的/ude1 山林/n 就/d 有/vyou 两/m 、/w 三千/m ，/w 但是/c 要/v 去/vf 周边/n 据点/n ，/w 只能/v 走/v ，/w 山里/s 没有/v 路/n ，/w 都/d 是/vshi 丛林/n 小道/n 。/w ”/w "
        #test_sent = "[一/d 站/vi 站/vi]/mq 被/pbei 选/v 为/p 总部/nis 的/ude1 秘书/nnt 。/w "
        #test_sent = "记者/nnt 见到/v 了/ule [第/m 一批/mq]/mq 学员/nnt 乃/v 蒙/b ，/w [一/d 站/vi 站/vi]/mq 护送/v 来到/v 这里/rzs ，/w 知识分子/nnd ，/w 被/pbei 选/v 为/p 总部/nis 的/ude1 秘书/nnt 。/w “/w 我/rr 在/p 丛林/n 里/f 走/v 了/ule 一个/mq 多/a 月/n ！/w 来到/v 这里/rzs 是/vshi 我/rr 的/ude1 福分/n 。/w 为什么/ryv 各/rz 民族/n 不能/v 平等/a ？/w 一起/s 奋斗/vi ，/w 用/p  光明/ntc ”/w ，/w 乃/v 蒙/b 激动/a 地/ude2 说/v 。/w "
        #test_sent = "“/w 深山老林/l 有/vyou 不少/m 南/b 掸邦/ns 军/n 士兵/nnt ，/w 总部/nis 周边/n 的/ude1 山林/n 就/d 有/vyou 两/m 、/w 三千/m ，/w 但是/c 要/v 去/vf 周边/n 据点/n ，/w 只能/v 走/v ，/w 山里/s 没有/v 路/n ，/w 都/d 是/vshi 丛林/n 小道/n 。/w ”/w "
        #test_sent = "[中央/n	人民/n	广播/vn	电台/n]/nt "
        test_sent = "国家/n  主席/n  江/nr  泽民/nr "
        print(test_sent)
        test = process_merge_word_label(test_sent)
        print(test)

        sents = split_doc2sents(test)
        for sent in sents:
            lsit_word_label = match_label(sent)
            print(sent)

    elif is_test == 2:
        list_file = [ 'G:/test_data/NLP/词性标注/人民日报语料库2014/1998_new/199801.txt']
        out_file = "G:/test_data/train_data"
        process_files(list_file, out_file)
    elif is_test == 1:
        data_dir = DATADIR
        list_name = []
        listdir(data_dir, list_name)

        all_len = len(list_name)
        if all_len <= 100:
            process_files2(list_name, RESULTS_DIR, 0.05, 1)
        else:
            train_list_len = all_len * 96 // 100
            testa_list_len = all_len * 2 // 100
            testb_list_len = all_len - train_list_len - testa_list_len

            print(train_list_len)
            print(testa_list_len)
            print(testb_list_len)

            train_list_file_name = list_name[0:train_list_len]
            r_len = train_list_len + 1
            testa_list_file_name = list_name[r_len:r_len + testa_list_len]
            r_len = train_list_len + testb_list_len + 1
            testb_list_file_name = list_name[r_len:r_len + testb_list_len - 2]

            #
            # listdir("G:/test_data/NLP/词性标注/人民日报语料库2014/1998_new", train_list_file_name)

            out_file = RESULTS_DIR + "/train_data"
            process_files(train_list_file_name, out_file)

            out_file = RESULTS_DIR + "/testa_data"
            process_files(testa_list_file_name, out_file)

            out_file = RESULTS_DIR + "/testb_data"
            process_files(testb_list_file_name, out_file)
