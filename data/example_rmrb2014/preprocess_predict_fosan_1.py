# 预测 预处理：
# 对佛山网数据进行预处理，给预测程序使用
# 1. 分句
# 2. 生成 （字 标签）对

g_word_split_label = ' '
g_label_other = "O"

g_sent_split = {
    "。": ["”", "＂"],
    "？": ["”", "＂"],
    "！": ["”", "＂"],
    "﹗": ["”", "＂"],
    "﹔": [],
    "；": []
    #"…": ["…"]
}
def __sent_end(doc, i, w_end):
    idx = i
    j = i + 1
    if j < len(doc):
        x = doc[j]
        if x in w_end:
            # 找到结束符
            idx = j
        else:
            # 自己就是结束符
            idx = i

    return idx

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
            cur_sent_end = i
            if len(g_sent_split[c]) > 0:  # 准结束符
                idx = __sent_end(doc, i, g_sent_split[c])
                if (idx > 0):  # 结束符结束
                    cur_sent_end = idx

            sent = doc[cur_sent_begin:cur_sent_end + 1]
            sents.append(sent)
            cur_sent_begin = cur_sent_end + 1
            i = cur_sent_end
        i += 1

    # +4 原因：最后只有一个字符例如 /n
    if cur_sent_begin + 2 < i:
        sents.append(doc[cur_sent_begin: i + 1])

    return sents



def __split_long_sent(sent,sent_list_out):
    max_num = 100
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
    max_num = 100
    if len(sent) >= max_num:
        pos = -1
        i = len(sent) // 2 - 10
        while i < len(sent):
            x = sent[i]
            if x in split_chars:
                pos = i
                break
            else:
               i += 1

        if pos == -1:
            split_chars2 = ["、"]
            pos = -1
            i = len(sent) // 2 - 10
            while i < len(sent):
                x = sent[i]
                if x in split_chars2:
                    pos = i
                    break
                else:
                    i += 1

        if pos != -1 and pos + 1 < len(sent):
            pos += 1
            sents = [sent[0:pos], sent[pos:]]

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

def match_label(sent):
    for word in sent:
        if word != ' ' and word != '　':
            yield word + g_word_split_label + g_label_other

def process_files(list_file, out_file):
    fw2 = open(out_file, 'w')
    fw2.close()

    for lf in list_file:
        with open(out_file, 'a+', encoding='utf-8') as fw:
            with open(lf) as fr:
                for line in fr:
                    doc = line.strip().split('\n') # process_merge_word_label(line)
                    for d in doc:
                        if len(d) > 0:
                            sents = split_doc2sents(d)
                            for sent in sents:
                                if len(sent) > 0:
                                    for word_tag in match_label(sent):
                                        fw.write('{}\n'.format(word_tag))

                                    fw.write('\n')


if __name__ == '__main__':
    predict_data_file_path = "G:/test_data/NLP/词性标注/佛山网/2018/1115.txt"
    predict_data_file_path_out = "G:/test_data/NLP/词性标注/佛山网/2018/1115_out.txt"
    process_files([predict_data_file_path], predict_data_file_path_out)

    #doc = "“以前生活环境总是受沙尘、污水以及噪音污染，非常痛苦，现在环境好很多了。”美的御海东郡小区物管和业主代表陈先生昨日在容桂高黎砂石场衷心地感谢市第三生态环境督察组，感谢容桂街道党工委、容桂街道办事处和相关职能部门还给业主们一个美好的生活环境，并现场送上锦旗。"
    #sents = split_doc2sents(doc)
    #print(sents)
