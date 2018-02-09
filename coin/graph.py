import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
import json
import os

np.random.seed(0)


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
f = []
start_date = 0
end_date = 0

cnt = 0
for file in files:
    if cnt == 0:
        start_date = int(file.split('.')[0])

    print(cnt, 'read:', file)

    dim = []
    with open(IN_PATH + file) as data_:
        dict = json.load(data_)
        dim.append(dict['volume'])
        dim.append(dict['close'])

    f.append(dim)

    end_date = int(file.split('.')[0])
    cnt += 1

len_sequences = cnt # 전체 시계열 길이

"""
visualization
"""


def two_scales(ax1, data1, data2, c1, c2):
    ax1.plot(data1, color=c1)
    ax1.set_xlabel('day')
    ax1.set_ylabel('volume')
    ax1.set_yscale('log')

    ax2 = ax1.twinx()
    ax2.plot(data2, color=c2)
    ax2.set_ylabel('price')
    ax2.set_yscale('linear')

    return ax1, ax2


s1 = [f[i][0] for i in range(len_sequences)]
s2 = [f[i][1] for i in range(len_sequences)]

# Create axes
fig, ax = plt.subplots()
ax1, ax2 = two_scales(ax, s1, s2, 'r', 'b')


# Change color of each axis
def color_y_axis(ax, color):
    """Color your axes."""
    for t in ax.get_yticklabels():
        t.set_color(color)
    return None


color_y_axis(ax1, 'r')
color_y_axis(ax2, 'b')
plt.show()
