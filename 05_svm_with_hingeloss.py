from pyspark.sql import DataFrame, SparkSession
from pyspark.ml.classification import LinearSVC
from pyspark.ml.linalg import Vectors
from operator import add
import re
import sys
import numpy as np
import time
from numpy import square, ones, array, dot, append, sqrt

if len(sys.argv) > 2:
    filepath1 = '/input/' + sys.argv[1]
    filepath2 = '/input/' + sys.argv[1]
    spark = SparkSession.builder.appName('bigdata').getOrCreate()
else:
    # run small data in local
    filepath = 'SmallTrainingData.txt'
    spark = (SparkSession.builder
             .appName('bigdata')
             .config('spark.driver.memory', '8g')
             .master('local[*]')
             .getOrCreate())
spark.sparkContext.setLogLevel("ERROR")
info_list = []


def pre_process(line):
    if '<doc id="AU' in line[0]:
        is_court = 1.0
    else:
        is_court = 0.0
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
    tfidfs = []
    n = corpus_count
    for word, idf in corpus_vocab.items():
        tf = float(tf_map.get(word, 0) / text_n)
        tfidf = float(tf * np.log((n + 1) / (idf + 1)))
        tfidfs.append(tfidf)

    return (*row, Vectors.dense(tfidfs))


corpus_count = None
corpus_vocab = None


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
                 .filter(lambda x: x[1] / n <= 0.8)
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

        result = (df.rdd
                  .map(lambda x: cal_tfidf(x, col_i))
                  .toDF(df.columns + self.output_col)
                  )
        return result


def prepare_data(dim=20000) -> (DataFrame, DataFrame):
    global corpus_count, corpus_vocab
    if len(sys.argv) <= 2:
        df = spark.read.text(filepath)
        df: DataFrame = df.rdd.map(lambda line: pre_process(line)).toDF(['text', 'label'])
        print(df.count())
        df.persist()
        model = Tfidf(input_col='text', output_col=['tfidfvector'], dim=dim)
        print('fitting...')
        corpus_count, corpus_vocab = model.fit(df)

        sc = spark.sparkContext
        sc.broadcast(corpus_vocab)
        print('transforming...')
        df = model.transform(df)
        # 线性回归
        train, test = df.randomSplit([9.0, 1.0], seed=100)
        train.persist()
        test.persist()
        df.unpersist()
        return train, test
    else:
        df = spark.read.text(filepath1)
        test = spark.read.text(filepath2)
        df: DataFrame = df.rdd.map(lambda line: pre_process(line)).toDF(['text', 'label'])
        df.persist()

        model = Tfidf(input_col='text', output_col=['tfidfvector'], dim=dim)
        print('fitting...')

        corpus_count, corpus_vocab = model.fit(df)

        sc = spark.sparkContext
        sc.broadcast(corpus_vocab)
        print('transforming...')
        train = model.transform(df)
        # 线性回归
        train.persist()
        df.unpersist()
        test = model.transform(test)
        test.persist()
        return train, test


def get_pred(v):
    if v >= 0.5:
        return 1.0
    else:
        return 0.0


def task1(train: DataFrame, test: DataFrame):
    print('task1 spark ml LinearSVC...')
    t = time.time()
    col2index = dict((col, index) for index, col in enumerate(train.columns))
    x_i, y_i = col2index.get('tfidfvector'), col2index.get('label')

    model = LinearSVC(featuresCol="tfidfvector", labelCol="label", predictionCol="prediction",
                      maxIter=100, regParam=0.0, threshold=0.0)
    lrModel = model.fit(train)
    print('training cost:{}'.format(time.time() - t))
    # Print the coefficients and intercepts for logistic regression with multinomial family
    t = time.time()
    theta = array(list(lrModel.coefficients) + [lrModel.intercept])
    m = (train.rdd
         .map(lambda row: (row[y_i], get_pred(float(dot(array(list(row[x_i]) + [1.0]), theta)))))
         .map(lambda x: array([1 if abs(x[0] - x[1]) < 0.001 and x[0] == 1.0 else 0,
                               1 if abs(x[0] - x[1]) >= 0.001 and x[0] == 1.0 else 0,
                               1 if abs(x[0] - x[1]) >= 0.001 and x[0] == 0.0 else 0,
                               1 if abs(x[0] - x[1]) < 0.001 and x[0] == 0.0 else 0]
                              )).reduce(add)
         )
    tp, fn, fp, tn = list(m)
    print('train confusing matrix:')
    print(np.array([[tp, fn],
                    [fp, tn]]))
    train_acc = (tp + tn) / sum(m)
    train_f1 = 2 * tp / (2 * tp + fp + fn)
    m = (test.rdd
         .map(lambda row: (row[y_i], get_pred(float(dot(array(list(row[x_i]) + [1.0]), theta)))))
         .map(lambda x: array([1 if abs(x[0] - x[1]) < 0.001 and x[0] == 1.0 else 0,
                               1 if abs(x[0] - x[1]) >= 0.001 and x[0] == 1.0 else 0,
                               1 if abs(x[0] - x[1]) >= 0.001 and x[0] == 0.0 else 0,
                               1 if abs(x[0] - x[1]) < 0.001 and x[0] == 0.0 else 0
                               ])).reduce(add)
         )
    tp, fn, fp, tn = list(m)
    print('test confusing matrix:')
    print(np.array([[tp, fn],
                    [fp, tn]]))
    test_acc = (tp + tn) / sum(m)
    test_f1 = 2 * tp / (2 * tp + fp + fn)
    print('train_acc:{},train_f1:{}'.format(train_acc, train_f1))
    print('test_acc:{},test_f1:{}'.format(test_acc, test_f1))
    print('testing cost:{}'.format(time.time() - t))
    return train, test


def l_func(x, y, theta):
    y = 1.0 if y == 1.0 else -1.0
    return max(0.0, 1.0 - y * dot(theta, array(list(x) + [1.0])))


def array_func(x, y, theta):
    y = 1.0 if y == 1.0 else -1.0
    if dot(theta, array(list(x) + [1.0])) < 1.0:
        return - y * array(list(x) + [1.0])
    else:
        return ones(len(x) + 1) * 0.0


def get_pred2(v):
    if v >= 0.0:
        return 1.0
    else:
        return 0.0


def task2_svm(train: DataFrame, test: DataFrame, dim):
    col2index = dict((col, index) for index, col in enumerate(train.columns))
    x_i, y_i = col2index.get('tfidfvector'), col2index.get('label')
    theta = np.ones(dim + 1) * 0.1
    theta_old = np.ones(dim + 1) * 0.1
    max_iter = 20
    loss = np.nan
    lr = 0.01
    n = train.count()
    p0 = 0
    for step in range(1, max_iter + 1):
        rs, loss_old = (train.rdd.map(lambda row: np.append(
            array_func(row[x_i], row[y_i], theta),
            l_func(row[x_i], row[y_i], theta)))
                        .reduce(add),
                        loss
                        )
        print('step:%3d,learning_rate:%.5f,loss:%.5f' % (step, lr, rs[-1] / n))
        p0, lr, update_param = rmsprop(p0, rs[:-1], lr, dim, rs[-1] / n, loss_old)

        if lr < 1e-6:
            print('early stop')
            break
        if update_param:
            partial, loss = rs[:-1], rs[-1] / n
            theta, theta_old = theta - partial * lr, theta
        else:
            theta = theta_old

    m = (train.rdd
         .map(lambda row: (row[y_i], get_pred2(float(dot(array(list(row[x_i]) + [1.0]), theta)))))
         .map(lambda x: array([1 if abs(x[0] - x[1]) < 0.001 and x[0] == 1.0 else 0,
                               1 if abs(x[0] - x[1]) >= 0.001 and x[0] == 1.0 else 0,
                               1 if abs(x[0] - x[1]) >= 0.001 and x[0] == 0.0 else 0,
                               1 if abs(x[0] - x[1]) < 0.001 and x[0] == 0.0 else 0
                               ])).reduce(add)
         )
    tp, fn, fp, tn = list(m)
    print('train confusing matrix:')
    print(np.array([[tp, fn],
                    [fp, tn]]))
    train_acc = (tp + tn) / (sum(m))
    train_f1 = 2 * tp / (2 * tp + fp + fn)
    m = (test.rdd
         .map(lambda row: (row[y_i], get_pred2(float(dot(array(list(row[x_i]) + [1.0]), theta)))))
         .map(lambda x: array([1 if abs(x[0] - x[1]) < 0.001 and x[0] == 1.0 else 0,
                               1 if abs(x[0] - x[1]) >= 0.001 and x[0] == 1.0 else 0,
                               1 if abs(x[0] - x[1]) >= 0.001 and x[0] == 0.0 else 0,
                               1 if abs(x[0] - x[1]) < 0.001 and x[0] == 0.0 else 0])).reduce(add)
         )
    tp, fn, fp, tn = list(m)
    print('test confusing matrix:')
    print(np.array([[tp, fn],
                    [fp, tn]]))
    test_acc = (tp + tn) / (sum(m))
    test_f1 = 2 * tp / (2 * tp + fp + fn)
    print('train_acc:{},train_f1:{}'.format(train_acc, train_f1))
    print('test_acc:{},test_f1:{}'.format(test_acc, test_f1))


def l_func_w(x, y, theta, ratio):
    y = 1.0 if y == 1.0 else -1.0
    ratio = ratio[0] if y == 1.0 else ratio[1]
    return max(0, 1 - y * dot(theta, array(list(x) + [1]))) / ratio


def array_func_w(x, y, theta, ratio):
    y = 1.0 if y == 1.0 else -1.0
    ratio = ratio[0] if y == 1.0 else ratio[1]
    if dot(theta, array(list(x) + [1])) < 1.0:
        return - y * array(list(x) + [1]) / ratio
    else:
        return ones(len(x) + 1) * 0.0


def task3_svm_with_weight_loss(train: DataFrame, test: DataFrame, dim):
    col2index = dict((col, index) for index, col in enumerate(train.columns))
    x_i, y_i = col2index.get('tfidfvector'), col2index.get('label')
    counts = train.rdd.map(lambda row: (row[y_i], 1)).reduceByKey(add).take(10)
    counts = dict((a, b) for a, b in counts)
    ratio = array([counts[1.0], counts[0.0]]) / (counts[0.0] + counts[1.0])
    print('ratio of train court and not court{}'.format(ratio))
    theta = np.ones(dim + 1) * 0.1
    theta_old = np.ones(dim + 1) * 0.1
    max_iter = 20
    loss = np.nan
    lr = 0.01
    n = train.count()
    print(n)
    p0 = 0
    for step in range(1, max_iter + 1):
        rs, loss_old = (train.rdd.map(lambda row: np.append(
            array_func_w(row[x_i], row[y_i], theta, ratio),
            l_func_w(row[x_i], row[y_i], theta, ratio)))
                        .reduce(add),
                        loss
                        )
        print('step:%3d,learning_rate:%.5f,loss:%.5f' % (step, lr, rs[-1] / n))
        p0, lr, update_param = rmsprop(p0, rs[:-1], lr, dim, rs[-1] / n, loss_old)

        if lr < 1e-6:
            print('early stop')
            break
        if update_param:
            partial, loss = rs[:-1], rs[-1] / n
            theta, theta_old = theta - partial * lr, theta
        else:
            theta = theta_old

    m = (train.rdd
         .map(lambda row: (row[y_i], get_pred2(float(dot(array(list(row[x_i]) + [1.0]), theta)))))
         .map(lambda x: array([1 if abs(x[0] - x[1]) < 0.001 and x[0] == 1.0 else 0,
                               1 if abs(x[0] - x[1]) >= 0.001 and x[0] == 1.0 else 0,
                               1 if abs(x[0] - x[1]) >= 0.001 and x[0] == 0.0 else 0,
                               1 if abs(x[0] - x[1]) < 0.001 and x[0] == 0.0 else 0
                               ])).reduce(add)
         )
    tp, fn, fp, tn = list(m)
    print('train confusing matrix:')
    print(np.array([[tp, fn],
                    [fp, tn]]))
    train_acc = (tp + tn) / (sum(m))
    train_f1 = 2 * tp / (2 * tp + fp + fn)

    m = (test.rdd
         .map(lambda row: (row[y_i], get_pred2(float(dot(array(list(row[x_i]) + [1.0]), theta)))))
         .map(lambda x: array([1 if abs(x[0] - x[1]) < 0.001 and x[0] == 1.0 else 0,
                               1 if abs(x[0] - x[1]) >= 0.001 and x[0] == 1.0 else 0,
                               1 if abs(x[0] - x[1]) >= 0.001 and x[0] == 0.0 else 0,
                               1 if abs(x[0] - x[1]) < 0.001 and x[0] == 0.0 else 0])).reduce(add)
         )
    tp, fn, fp, tn = list(m)
    print('test confusing matrix:')
    print(np.array([[tp, fn],
                    [fp, tn]]))
    test_acc = (tp + tn) / (sum(m))
    test_f1 = 2 * tp / (2 * tp + fp + fn)
    print('train_acc:{},train_f1:{}'.format(train_acc, train_f1))
    print('test_acc:{},test_f1:{}'.format(test_acc, test_f1))


def rmsprop(p0, p, learning_rate, dim, loss, loss_old):
    if loss_old - loss < 0:
        if loss / loss_old > 1.05:
            return p0, learning_rate * 0.5, 0
        else:
            return p0, learning_rate * 0.75, 1
    if 0 <= (loss_old - loss) / loss_old < 0.01:
        return p0, learning_rate * 1.1, 1
    p0 += 0.9 * p0 + 0.1 * sqrt(dot(p, p) / dim) + 1e-8
    learning_rate /= p0
    return p0, learning_rate, 1


def main():
    t = time.time()
    dim = 2000
    train, test = prepare_data(dim=dim)
    print('reading cost:{}'.format(time.time() - t))
    task1(train, test)
    task2_svm(train, test, dim=dim)
    task3_svm_with_weight_loss(train, test, dim=dim)


if __name__ == '__main__':
    main()
