import bz2
# 解压超大压缩文件
filepath = '/mnt/taxi-data-sorted-large.csv.bz2'
print(filepath)
new_filepath = '/mnt1/taxi-data-sorted-large.csv'
with open(new_filepath, 'wb') as nf, bz2.BZ2File(filepath, 'rb') as f:
    count = 0
    for data in iter(lambda: f.read(900 * 1024), b''):
        nf.write(data)
        count += 1
        if count % 1000 == 0:
            print(int(count * 900 / 1024))
print('depressed')
