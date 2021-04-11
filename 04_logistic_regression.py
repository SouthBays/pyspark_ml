from pyspark.sql import DataFrame, SparkSession
from operator import add
import re
import sys
import numpy as np
from numpy import array, dot

if len(sys.argv) > 2:
    filepath1 = '/input/' + sys.argv[1]
    filepath2 = '/input/' + sys.argv[2]
    spark = SparkSession.builder.appName('bigdata').getOrCreate()
else:
    # run small data in local
    filepath = 'SmallTrainingData.txt'
    spark = SparkSession.builder.appName('bigdata').config('spark.driver.memory', '8g').master('local[*]').getOrCreate()
info_list = []


def pre_process(line):
    if '<doc id="AU' in line[0]:
        is_court = '1'
    else:
        is_court = '0'
    line = re.findall(r'<doc id.*?">(.*)</doc>', line[0], flags=re.S)[0]
    return line, is_court


def cal_tfidf(row, input_index):
    text = row[input_index]
    tf_map = {}
    text_n = 0
    for word in text.split(' '):
        text_n += 1
        if word in tf_map.keys():
            tf_map[word] += 1
        else:
            tf_map[word] = 1
    tfs, tfidfs = [], []
    n = corpus_count
    for word, idf in corpus_vocab.items():
        tf = float(tf_map.get(word, 0) / text_n)
        tfidf = float(tf * np.log((n + 1) / (idf + 1)))
        tfs.append(tf)
        tfidfs.append(tfidf)

    return (*row, tfs, tfidfs)


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
        vocab = (self.corpus.rdd.flatMap(lambda x: list(set(x[0].split(' '))))
                 .map(lambda x: (x, 1))
                 .reduceByKey(add)
                 .filter(lambda x: x[0] in ["applicant", "and", "attack", "protein", "court"] or x[1] != n)
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
        flag = 0
        for col_i, col in enumerate(self.corpus.columns):
            if col == self.input_col:
                flag = 1
                break
        if not flag:
            raise Exception(f'input col{self.input_col} not exists in input dataframe!')
        # sc = spark.sparkContext
        # sc.broadcast(self.vocab)
        result = (df.rdd
                  .map(lambda x: cal_tfidf(x, col_i))
                  .toDF(df.columns + self.output_col)
                  )
        return result


def print_tf(df: DataFrame, n):
    print_list = ["applicant", "and", "attack", "protein", "court"]
    index_list = [corpus_vocab2index.get(x) for x in print_list]

    a, b = 0, 0
    aver1, aver0 = np.zeros(5), np.zeros(5)
    for row in df.collect():
        if (a + b) > 2 * n:
            break
        if row['label'] == '1':
            if a >= n:
                continue
            else:
                a += 1
                aver1 += np.array([row['tfvector'][index]
                                   if index is not None else 0
                                   for index in index_list])
        else:
            if b >= n:
                continue
            else:
                b += 1
                aver0 += np.array([row['tfvector'][index]
                                   if index is not None else 0
                                   for index in index_list])
    print('words', print_list)
    print('indexs', index_list)
    print(f'is_court size:{a}', list(aver1 / a))
    print(f'not_court size:{b}', list(aver0 / b))


def task1(dim=20000, is_print_tf=1) -> (DataFrame, DataFrame):
    global corpus_count, corpus_vocab, corpus_vocab2index
    if len(sys.argv) <= 1:
        df = spark.read.text(filepath)
        df: DataFrame = df.rdd.map(lambda line: pre_process(line)).toDF(['text', 'label'])
        print(df.count())
        df.persist()
        model = Tfidf(input_col='text', output_col=['tfvector', 'tfidfvector'], dim=dim)
        print('fitting...')

        corpus_count, corpus_vocab = model.fit(df)
        corpus_vocab2index = dict((x, index) for index, x in enumerate(corpus_vocab.keys()))

        sc = spark.sparkContext
        sc.broadcast(corpus_vocab)
        print('transforming...')
        df = model.transform(df)
        print('collecting...')
        if is_print_tf:
            print_tf(df, n=40)
        # 线性回归
        train, test = df.randomSplit([9.0, 1.0], seed=100)
        train.persist()
        test.persist()
        df.unpersist()
    else:
        df = spark.read.text(filepath1)
        test = spark.read.text(filepath2)
        df: DataFrame = df.rdd.map(lambda line: pre_process(line)).toDF(['text', 'label'])
        print(df.count())
        df.persist()
        model = Tfidf(input_col='text', output_col=['tfvector', 'tfidfvector'], dim=dim)
        print('fitting...')
        corpus_count, corpus_vocab = model.fit(df)
        corpus_vocab2index = dict((x, index) for index, x in enumerate(corpus_vocab.keys()))

        sc = spark.sparkContext
        sc.broadcast(corpus_vocab)
        print('transforming...')
        df = model.transform(df)
        print('collecting...')
        if is_print_tf:
            print_tf(df, n=40)
        # 线性回归
        df = df.repartition(20)
        df.persist()
        test: DataFrame = model.transform(test)
        test = test.repartition(20)
        test.persist()

    return df, test


def bold_driver(loss, loss_old, learning_rate):
    if loss > loss_old:
        learning_rate *= 0.75
    else:
        learning_rate *= 1.1
    return learning_rate


def loss_func(x, y, theta):
    y = float(y)
    h = 1 / (1 + np.power(np.e, -dot(theta, x)))
    loss = -y * np.log(h) - (1 - y) * np.log(1 - h)
    return loss


def grad_func(x, y, theta):
    y = float(y)
    h = 1 / (1 + np.power(np.e, -dot(theta, x)))
    partial = h - x * y
    return partial


def task2_3_train_evaluate(df: DataFrame, test: DataFrame, dim=20000):
    n = df.count()
    p = dim + 1
    theta = np.ones(p) * 0.1
    max_iter = 20
    learning_rate = 0.1
    lamb = 0.01
    tolerance = 0.01

    col2index = dict((col, index) for index, col in enumerate(df.columns))
    x_index, y_index = col2index.get('tfidfvector'), col2index.get('label')
    print(x_index, y_index)
    loss = (df.rdd
            .map(lambda row: loss_func(array(row[x_index] + [1.0]),
                                       row[y_index], theta)).sum() / n
            + dot(theta, theta) * lamb / 2 / n
            )
    info = 'initial,loss:%.5f,lr:%.5f' % (loss, learning_rate)
    print(info)
    info_list.append(info)
    c = 0
    for step in range(max_iter):
        # cal gradient descent
        partial = (df.rdd
                   .map(lambda row: grad_func(array(row[x_index] + [1.0]),
                                              row[y_index], theta))
                   .reduce(add) / n
                   + lamb / n * theta
                   )
        # apply the gradient descent to params
        theta -= partial * learning_rate
        loss, loss_old = (df.rdd
                          .map(lambda row: loss_func(array(row[x_index] + [1]),
                                                     row[y_index], theta)).sum() / n
                          + dot(theta, theta) * lamb / 2 / n,
                          loss
                          )
        info = 'step:%2d,loss:%.5f,lr:%.5f' % (step, loss, learning_rate)
        print(info)
        info_list.append(info)
        # use bold driver to adjust learning rate
        learning_rate = bold_driver(loss, loss_old, learning_rate)
        if loss - loss_old < tolerance:
            c += 1
        if learning_rate < tolerance:
            info = 'early stop'
            print(info)
            info_list.append(info)
            break
    print('evaluating...')
    threshhold = 0.5
    train_acc = (df.rdd
                 .map(lambda row: (row[y_index], get_label(float(dot(array(row[x_index] + [1.0]), theta)))))
                 .filter(lambda x: x[0] == x[1]).count()
                 ) / n
    test_acc = (test.rdd
                .map(lambda row: (row[y_index], get_label(float(dot(array(row[x_index] + [1.0]), theta)))))
                .filter(lambda x: x[0] == x[1]).count()
                ) / test.count()
    print('train_acc:{},test_acc:{}'.format(train_acc, test_acc))
    return theta


def get_label(v):
    if v >= 0.5:
        return '1'
    else:
        return '0'


def main():
    train, test = task1(dim=2000, is_print_tf=1)
    task2_3_train_evaluate(train, test, dim=2000)


if __name__ == '__main__':
    main()
