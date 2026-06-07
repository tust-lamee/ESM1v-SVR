# ESM1v-SVR
The ESM-1v protein language model was combined with Support Vector Regression (SVR) to develop a prediction framework for the optimal temperature of enzymes.

9581文件为本研究的蛋白质序列，文件格式为 蛋白序列 标签     
1.先提取9581条蛋白质序列的特征向量（使用ESM-1v.py）

2.跑SVR.py得到本研究所使用的.pkl
3.跑ESM-1v.py得到未知蛋白序列的特征向量
4.跑predictied_temperatures.py，输出未知蛋白序列的预测值 .csv
