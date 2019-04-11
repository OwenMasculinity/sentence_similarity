# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import jieba
import math
import distance
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from scipy.linalg import norm
import gensim



def cosine_similarity(s1, s2):
    # 计算关键字
    def KeyWords(line1, line2):
        word_freq = {}
        word_freq2 = {}
        # 分词
        words1 = jieba.cut(line1, cut_all=False)
        words2 = jieba.cut(line2, cut_all=False)
        # 得出第一列的关键词
        for word in words1:
            word_freq[word] = 1
        freq_word = []
        for word, freq in word_freq.items():
            freq_word.append((word, freq))
            # 得出第二列的关键词
        for word2 in words2:
            word_freq2[word2] = 1
        freq_word2 = []
        for word2, freq2 in word_freq2.items():
            freq_word2.append((word2, freq2))
        return freq_word, freq_word2

    # 统计关键词及个数 并计算相似度
    def MergeKeys(freq_word, freq_word2):
        # 合并关键词 采用三个数组实现
        arrayKey = []
        for i in range(len(freq_word)):
            arrayKey.append(freq_word[i][0])  # 向数组中添加元素
        for i in range(len(freq_word2)):
            if freq_word2[i][0] in arrayKey:
                pass
            else:  # 合并
                arrayKey.append(freq_word2[i][0])

                # 计算词频 infobox可忽略TF-IDF
        arrayNum1 = [0] * len(arrayKey)
        arrayNum2 = [0] * len(arrayKey)

        # 赋值arrayNum1
        for i in range(len(freq_word)):
            key = freq_word[i][0]
            value = freq_word[i][1]
            j = 0
            while j < len(arrayKey):
                if key == arrayKey[j]:
                    arrayNum1[j] = value
                    break
                else:
                    j = j + 1

                    # 赋值arrayNum2
        for i in range(len(freq_word2)):
            key = freq_word2[i][0]
            value = freq_word2[i][1]
            j = 0
            while j < len(arrayKey):
                if key == arrayKey[j]:
                    arrayNum2[j] = value
                    break
                else:
                    j = j + 1

        # print arrayNum1
        # print arrayNum2
        # print len(arrayNum1),len(arrayNum2),len(arrayKey)

        # 计算两个向量的点积
        x = 0
        i = 0
        while i < len(arrayKey):
            x = x + arrayNum1[i] * arrayNum2[i]
            i = i + 1
            # print x

        # 计算两个向量的模
        i = 0
        sq1 = 0
        while i < len(arrayKey):
            sq1 = sq1 + arrayNum1[i] * arrayNum1[i]  # pow(a,2)
            i = i + 1
            # print sq1

        i = 0
        sq2 = 0
        while i < len(arrayKey):
            sq2 = sq2 + arrayNum2[i] * arrayNum2[i]
            i = i + 1
            # print sq2

        result = float(x) / (math.sqrt(sq1) * math.sqrt(sq2))
        return result

    freq_word, freq_word2 = KeyWords(sen1, sen2)
    result = MergeKeys(freq_word, freq_word2)

    return result


def jaccard_similarity(s1, s2):
    def add_space(s):
        return ' '.join(list(s))

    # 将字中间加入空格
    s1, s2 = add_space(s1), add_space(s2)
    # 转化为TF矩阵
    cv = CountVectorizer(tokenizer=lambda s: s.split())
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()
    # 求交集
    numerator = np.sum(np.min(vectors, axis=0))
    # 求并集
    denominator = np.sum(np.max(vectors, axis=0))
    # 计算杰卡德系数
    return 1.0 * numerator / denominator


def tf_similarity(s1, s2):
    def add_space(s):
        return ' '.join(list(s))

    # 将字中间加入空格
    s1, s2 = add_space(s1), add_space(s2)
    # 转化为TF矩阵
    cv = CountVectorizer(tokenizer=lambda s: s.split())
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()
    # 计算TF系数
    return np.dot(vectors[0], vectors[1]) / (norm(vectors[0]) * norm(vectors[1]))


def tfidf_similarity(s1, s2):
    def add_space(s):
        return ' '.join(list(s))

    # 将字中间加入空格
    s1, s2 = add_space(s1), add_space(s2)
    # 转化为TF矩阵
    cv = TfidfVectorizer(tokenizer=lambda s: s.split())
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()
    # 计算TF系数
    return np.dot(vectors[0], vectors[1]) / (norm(vectors[0]) * norm(vectors[1]))


"""
对分好的每一个词获取其对应的Word Vector，然后将所有 Vector 相加并求平均，这样就可得到 Sentence Vector 了
"""
def vector_similarity1(s1, s2):
    model_file = './word2vec/news_12g_baidubaike_20g_novel_90g_embedding_64.bin'
    model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=True)
    def sentence_vector(s):
        words = jieba.lcut(s)
        v = np.zeros(64)
        for word in words:
            v += model[word]
        v /= len(words)
        return v
    v1, v2 = sentence_vector(s1), sentence_vector(s2)
    return np.dot(v1, v2) / (norm(v1) * norm(v2))

def vector_similarity2(s1, s2):
    model_file = './word2vec/news_sohusite_300.bin'
    model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=True)
    def sentence_vector(s):
        words = jieba.lcut(s)
        v = np.zeros(300)
        for word in words:
            v += model[word]
        v /= len(words)
        return v
    v1, v2 = sentence_vector(s1), sentence_vector(s2)
    return np.dot(v1, v2) / (norm(v1) * norm(v2))


def main_similartiy(sen1,sen2,method="cosine"):
    if method=="cosine": # 根据词向量（one-hot），计算余弦夹角
        result = cosine_similarity(sen1, sen2)
    elif method=="edit_distance": #  Levenshtein 距离(编辑距离): 两个字串之间，由一个转成另一个所需的最少编辑操作次数
        result = distance.levenshtein(sen1, sen2)
    elif method=="jaccard_index": # 杰卡德系数: 两个样本的交集除以并集得到的数值
        result = jaccard_similarity(sen1, sen2)
    elif method=="tf_index": #  TF 系数
        result = tf_similarity(sen1, sen2)
    elif method=="tfidf_index": # TFIDF 计算
        result = tfidf_similarity(sen1, sen2)
    elif method=="sentence_vector1":
        result = vector_similarity1(sen1, sen2)
    elif method=="sentence_vector2":
        result = vector_similarity2(sen1, sen2)

    return result

if __name__=="__main__":
    sen1="我在上海工作了一段时间"
    sen2="我在上海陆家嘴工作的时间有三个月"
    result = main_similartiy(sen1, sen2, method="sentence_vector2")
    print(result)