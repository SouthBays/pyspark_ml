# 安装依赖  
pip install -r requirements.txt  
# win10运行  
需要配置环境变量HADOOP_HOME,bin文件夹下winutils.exe  
支持文件相对路径
```python
import os
os.environ['HADOOP_HOME']='./hadoop'
```
# linux运行文件路径设置  
+ 本地路径：file://+绝对路径,如file:///home/file.txt  
+ 绝对路径：spark会认为是hdfs地址需要配置hadoop环境    
+ 相对路径：会报错 

[一、tfidf](02_tfidf.py)  
[二、linear_regression](03_linear_regression.py)  
[三、logistic_regression](04_logistic_regression.py)  
[四、svm_with_hingeloss](05_svm_with_hingeloss.py)  





