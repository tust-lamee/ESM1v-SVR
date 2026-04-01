# ESM1v-SVR
The ESM-1v protein language model was combined with Support Vector Regression (SVR) to develop a prediction framework for the optimal temperature of enzymes.

1.先跑ESM-1v.py得到特征特征向量
2.跑SVR.py得到所需要的.pkl
3.跑ESM1V-SVR.py，未知蛋白序列以fasta格式输入，直接输出预测值 .csv
