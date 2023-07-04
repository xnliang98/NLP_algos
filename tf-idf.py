import math
from collections import defaultdict
 

# 词频统计
def Counter(words):
    word_count = []
    for sentence in words:
        word_dict = defaultdict(int)
        for word in sentence:
            word_dict[word] += 1
        word_count.append(word_dict)
    return word_count

corpus = [
    "what is the weather like today",
    "what is for dinner tonight",
    "this is a question worth pondering",
    "it is a beautiful day today"
]
words = []
# 分词
for sentence in corpus:
    words.append(sentence.strip().split())
word_count = Counter(words)

def TF(word, word_dict):
    return word_dict[word] / sum(word_dict.values())

def count_doc(word, word_count):
    cnt = 0
    for word_dict in word_count:
        if word in word_dict:
            cnt += 1
    return cnt

def IDF(word, word_count):
    return math.log(len(word_count) / (count_doc(word, word_count) + 1)) # 不能除0

def TF_IDF(word, word_dict, word_count):
    # $$ word_freq / num_words_in_doc * log(num_doc / (1 + num_doc_cotains_word)) $$
    return TF(word, word_dict) * IDF(word, word_count)

p = 1
for word_dict in word_count:
    print("part:{}".format(p))
    p += 1
    for word, cnt in word_dict.items():
        print("word: {} ---- TF-IDF:{}".format(word, TF_IDF(word, word_dict, word_count)))
