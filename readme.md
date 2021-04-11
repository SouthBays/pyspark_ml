# win10运行需要适配的winutil.exe
```python
import os
os.environ['HADOOP_HOME']='./hadoop'
```
pip install -r requirements.txt
[一、tfidf](02_tfidf.py)  
[二、linear_regression](03_linear_regression.py)  
[三、logistic_regression](04_logistic_regression.py)  
[四、svm_with_hingeloss](05_svm_with_hingeloss.py)  
# 注  
win10下文件路径可以为相对路径。
mac或者linux下必须为
file://+绝对路径,如file:///home/file.txt  
如果为绝对路径或者相对路径会有问题  



