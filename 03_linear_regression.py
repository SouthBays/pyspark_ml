import numpy as np
from numpy import square, ones, array, dot
import matplotlib.pyplot as plt
from pyspark.sql import DataFrame, SparkSession
import sys
if len(sys.argv) > 1:
    filepath = '/input/' + sys.argv[1]
    spark = SparkSession.builder.appName('bigdata').getOrCreate()
else:
    # run small data in local
    filepath = 'taxi-data-sorted-small.csv'
    spark = SparkSession.builder.appName('bigdata').config('spark.driver.memory', '4g').master('local[*]').getOrCreate()
info_list = []
# dataset amount
n = None


def filt_func(x):
    try:
        if 120 <= float(x[4]) <= 3600 and 3 <= float(x[11]) <= 200 and 1 <= float(x[5]) <= 50:
            return True
        else:
            return False
    except:
        # empty str that cannot convert to float
        return False


def func_plus(x, y):
    return x + y


def load_data(filename, numpartition=10, show=10, limit=None) -> DataFrame:
    fields_dict = {4: 'trip_time_in_secs',
                   5: 'trip_distance',
                   11: 'fare_amount',
                   12: 'tolls_amount',
                   16: 'total_amount'}
    print('=' * 20)
    print('start reading csv')
    print('=' * 20)
    df = spark.read.options(header=False, sep=',').csv(filename)
    # if Remove all taxi rides that have ”tolls amount” less than 3 dollar, there's 0 out of 200w data
    # so i do not apply this condition in filt func
    print('=' * 20)
    print('filtering')
    print('=' * 20)

    df = df.rdd.filter(filt_func) \
        .map(lambda x: tuple(float(x[k]) for k in fields_dict.keys())) \
        .toDF(list(fields_dict.values()))
    if limit and type(limit) == int:
        df = df.limit(limit)
    print('=========repartition=========')
    df = df.repartition(numpartition)
    df.persist()
    print('=' * 20)
    global n
    n = df.count()
    print(f'total:{n}, num partions:{df.rdd.getNumPartitions()}, show top:{show}')
    print('=' * 20)
    df.show(show)
    return df


def task1_simple_lr(df: DataFrame):
    df.createTempView("taxi_rides")
    # calculate in spark sql
    result = spark.sql("""select
    ((n*sum_xy-sum_x*sum_y)/(n*sum_x_2-sum_x*sum_x)) as m,
    ((sum_x_2*sum_y-sum_x*sum_xy)/(n*sum_x_2-sum_x*sum_x)) as b
    from (select {} as n,
    sum(trip_distance * fare_amount) as sum_xy,
    sum(trip_distance) as sum_x,
    sum(fare_amount) as sum_y,
    sum(trip_distance * trip_distance) as sum_x_2
    from taxi_rides) params""".format(n))
    # cache results
    result.cache()
    # show the params of m and b
    result.show()
    m, b = 0.0, 0.0
    for row in result.collect():
        m = row['m']
        b = row['b']
    info_list.append('task 1==m:{},b:{}'.format(m, b))
    x, y = [], []
    for row_i, row in enumerate(df.collect()):
        x.append(row['trip_distance'])
        y.append(row['fare_amount'])
        if row_i >= 1000:
            break
    max_x = max(x)
    min_x = min(x)
    # plot the points and line
    plt.scatter(x, y, marker='.')
    x = np.linspace(min_x, max_x, num=int((max_x - min_x) / 0.1))
    y = m * x + b
    plt.plot(x, y, label=f'y={m}*x + {b}')
    plt.title('first 1000 points and line of simple lr cal by million of points')
    plt.xlabel('trip_distance')
    plt.ylabel('fare_amount')
    plt.show()
    return m, b

def bold_driver(loss, loss_old, learning_rate):
    if loss > loss_old:
        learning_rate *= 0.5
    else:
        learning_rate *= 1.05
    return learning_rate


def task2_simple_lr(df: DataFrame, baseline_m, baseline_b):
    m, b = 0.1, 0.1
    max_iter = 100
    learning_rate = 0.01
    loss = df.rdd.map(lambda row: square(row[2] - baseline_m * row[1] - baseline_b)).sum() / n
    print('baseline,loss:%.5f,m:%.5f,b:%.5f' % (loss, baseline_m, baseline_b))
    info_list.append('baseline,loss:%.5f,m:%.5f,b:%.5f' % (loss, baseline_m, baseline_b))
    loss = df.rdd.map(lambda row: square(row[2] - m * row[1] - b)).sum() / n
    print('initial,loss:%.5f,m:%.5f,b:%.5f' % (loss, m, b))
    info_list.append('initial,loss:%.5f,m:%.5f,b:%.5f' % (loss, m, b))
    for step in range(max_iter):
        partial = df.rdd.map(lambda row: array([- row[1] * (row[2] - (m * row[1] + b)),
                                                - (row[2] - (m * row[1] + b))])
                             ).aggregate(array([0.0, 0.0]), func_plus, func_plus)

        m -= partial[0] * learning_rate
        b -= partial[1] * learning_rate
        loss, loss_old = df.rdd.map(lambda row: square(row[2] - m * row[1] - b)
                                    ).aggregate(0, func_plus, func_plus) / n, loss
        info = 'step:%3d,loss:%.5f,m:%.5f,b:%.5f,learning_rate:%.5f' % (step, loss, m, b, learning_rate)
        print(info)
        info_list.append(info)
        learning_rate = bold_driver(loss, loss_old, learning_rate)


def task3_multi_lr(df: DataFrame):
    # parmas：time_in_secs,distance,fare,tolls,const
    theta = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
    max_iter = 100
    learning_rate = 0.001
    tolerance = 0.00001
    loss = df.rdd.map(lambda row: square(row[4] - dot(theta, array([row[0], row[1], row[2], row[3], 1])))).sum() / n
    info = 'initial,loss:%.5f,time_in_secs:%.5f,distance:%.5f,fare:%.5f,tolls:%.5f,const:%.5f,lr:%.5f' % (
        loss, theta[0], theta[1], theta[2], theta[3], theta[4], learning_rate)
    print(info)
    info_list.append(info)
    c = 0
    for step in range(max_iter):
        # cal gradient descent
        partial = df.rdd.map(lambda row:
                             - theta * (ones(5) * (row[4] - dot(theta, array([row[0], row[1], row[2], row[3], 1]))))
                             ).aggregate(np.zeros(5), func_plus, func_plus) * 2 / n
        # apply the gradient descent to params
        theta -= partial * learning_rate
        loss, loss_old = df.rdd.map(lambda row: square(
            row[4] - dot(theta, array([row[0], row[1], row[2], row[3], 1])))
                                    ).aggregate(0, func_plus, func_plus) / n, loss
        info = 'step:%3d,loss:%.5f,time_in_secs:%.5f,distance:%.5f,fare:%.5f,tolls:%.5f,const:%.5f,lr:%.5f' % (
            step + 1, loss, theta[0], theta[1], theta[2], theta[3], theta[4], learning_rate)
        print(info)
        info_list.append(info)
        # use bold driver to adjust learning rate
        learning_rate = bold_driver(loss, loss_old, learning_rate)
        if loss - loss_old < tolerance:
            c += 1
        if learning_rate < tolerance or c >= 10:
            info = 'early stop'
            print(info)
            info_list.append(info)
            break


def main():
    df = load_data(filepath, numpartition=100, show=10, limit=None)
    baseline_m, baseline_b = task1_simple_lr(df)
    task2_simple_lr(df, baseline_m, baseline_b)
    task3_multi_lr(df)
    [print(k) for k in info_list]


if __name__ == '__main__':
    main()
