import numpy as np
import matplotlib.pyplot as plt
import json
import os


"""
JSON 파일 추출
"""
files = []
IN_PATH = './result/'
for filename in os.listdir(IN_PATH):
    if filename.endswith(".json"):
        files.append(filename)

"""
데이터 읽기
"""
prices = []
for file in files:
    print('read:', file)
    with open(IN_PATH + file) as data_:
        data = json.load(data_)
    prices.append(data['weightedAverage'])

X = np.array(prices)
print(X)

'''
visualization
'''
plt.rc('font', family='serif')
plt.figure()
plt.plot(X)
plt.show()
