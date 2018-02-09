import os
import json
import csv
from datetime import datetime


"""
JSON 파일 추출
"""
files = []
PATH = './result/'
for filename in os.listdir(PATH):
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

    # print(cnt, 'read:', file)

    with open(PATH + file) as data_:
        dict = json.load(data_)
        del dict['quoteVolume']
        del dict['weightedAverage']

    f.append(dict)

    end_date = int(file.split('.')[0])
    cnt += 1

print("complete: READ")

"""
csv 쓰기
"""
output = open(PATH + 'output.csv', 'w', newline='')
wt = csv.writer(output)
wt.writerow(['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])

for row in f:
    # Date
    date_ = datetime.fromtimestamp(row['date'])
    year_ = str(date_.year)
    month_ = str(date_.month)
    if len(month_) < 2:
        month_ = '0' + month_
    day_ = str(date_.day)
    if len(day_) < 2:
        day_ = '0' + day_
    date__ = year_ + month_ + day_

    wt.writerow([date__, row['open'], row['high'], row['low'], row['close'], row['volume']])

output.close()
print("complete: WRITE")
