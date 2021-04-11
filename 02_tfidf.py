from pyspark.sql import DataFrame, SparkSession
from pyspark.ml.linalg import Vectors, DenseVector
from operator import add
import re
import sys
import numpy as np
import time
from numpy import square, ones, array, dot, append

if len(sys.argv) > 1:
    filepath1 = '/input/' + sys.argv[1]
    spark = SparkSession.builder.appName('bigdata').getOrCreate()
else:
    # run small data in local
    filepath = 'WikipediaPagesOneDocPerLine1000LinesSmall.txt'
    spark = SparkSession.builder.appName('bigdata').config('spark.driver.memory', '6g').master('local[*]').getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
info_list = []


def pre_process(line):
    a = re.findall(r'<doc id="(.*?)"', line, flags=re.S)[0]
    b = re.findall(r'">(.*?)</doc>', line, flags=re.S)[0]

    return a, b


def cal_tfidfs(text):
    tf_map = {}
    text_n = 0
    for word in text.split(' '):
        text_n += 1
        if word in tf_map.keys():
            tf_map[word] += 1
        else:
            tf_map[word] = 1
    tfidfs = []
    n = corpus_count
    for word, idf in corpus_vocab.items():
        tf = float(tf_map.get(word, 0) / text_n)
        tfidf = float(tf * np.log((n + 1) / (idf + 1)))
        tfidfs.append(tfidf)
    return tfidfs


def cal_tfidf(row, input_index):
    text = row[input_index]
    tfidfs = cal_tfidfs(text)

    return (*row, Vectors.dense(tfidfs))


corpus_count = None
corpus_vocab = None
corpus_vocab2index = None


class Tfidf:
    corpus = None
    tfidf_map = None
    input_col = None
    output_col = None

    def __init__(self, input_col='text', output_col='vector', dim=100):
        self.input_col = input_col
        self.output_col = output_col
        self.dim = dim

    def get_vocab(self, n):
        print('getting corpus...')
        vocab = (self.corpus.rdd.flatMap(lambda x: list(set(x[1].split(' '))))
                 .map(lambda x: (x, 1))
                 .reduceByKey(add)
                 .filter(lambda x: x[1] != n)
                 .sortBy(lambda x: -x[1])
                 )
        vocab = vocab.take(self.dim)
        vocab = dict((word[0], word[1]) for word in vocab)
        print('vocab dim', len(vocab.keys()))
        return vocab

    def fit(self, corpus: DataFrame):
        self.corpus = corpus
        n = self.corpus.count()
        return n, self.get_vocab(n)

    def transform(self, df: DataFrame):
        col2index = dict((col, index) for index, col in enumerate(df.columns))
        col_i = col2index[self.input_col]
        result = (df.rdd
                  .map(lambda x: cal_tfidf(x, col_i))
                  .toDF(df.columns + self.output_col)
                  )
        return result

    def get_prediction(self, df: DataFrame, text, topK):
        vector = Vectors.dense(cal_tfidfs(text))
        res = df.rdd.map(lambda row: (*row, vector.dot(row[-1]))).sortBy(lambda x: -x[-1]).take(topK)
        [print(row[0], row[-1], row[1][:200]) for row in res]


def prepare_data(dim=20000) -> (DataFrame, DataFrame):
    global corpus_count, corpus_vocab, corpus_vocab2index
    df = spark.read.text(filepath)

    df: DataFrame = df.rdd.map(lambda line: pre_process(line[0])).toDF(['id', 'text'])
    df.persist()
    print('line count', df.count())
    model = Tfidf(input_col='text', output_col=['tfidfvector'], dim=dim)
    print('fitting...')
    corpus_count, corpus_vocab = model.fit(df)
    corpus_vocab2index = dict((x, index) for index, x in enumerate(corpus_vocab.keys()))

    sc = spark.sparkContext
    sc.broadcast(corpus_vocab)
    print('transforming...')
    res = model.transform(df)
    res.persist()
    df.unpersist()
    return model, res


def main():
    dim = 2000
    model, df = prepare_data(dim=dim)
    text = """Norwegian Military AcademyThe Norwegian Military Academy ("Krigsskolen"), in Oslo, educates officers of the Norwegian Army"""
    model.get_prediction(df, text, topK=20)


if __name__ == '__main__':
    main()
