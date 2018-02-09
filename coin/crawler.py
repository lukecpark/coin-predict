"""
유닉스 시간 (Unix Time)이란?
세계 표준시 (UTC) 로 1970년 1월 1일 00시:00분:00초를
기준으로 하여 지금 현재까지 흐른 모든 시간을 초(sec) 단위로 표현한 것입니다.
단, 윤초(Leap Second)는 반영하지 않습니다.

ref. http://mwultong.blogspot.com/2006/12/python-get-unix-time-epoch.html
"""

"""
date: 해당 tic이 시작하는 timestamp
high: 해당 tic의 최고점
low: 해당 tic의 최저점
open: 해당 tic이 시작할 때의 가격
close: 해당 tic이 끝날 때 (혹은 해당 tic의 가장 마지막에 거래된) 가격
Volume: 해당 tic의 거래량(기준통화 (USDT) 단위)
quoteVolume: 해당 tic의 거래량 (거래 대상 통화 (BTC) 단위)
weightedAverage: 해당 tic의 거래량이 반영된 평균 거래가격

ref. https://steemit.com/kr/@axiomier/poloniex-hts-1-api
"""


from urllib.request import urlopen
from time import time
import json
from pprint import pprint
from time import sleep
import os


"""
출력경로 지정
"""
OUT_PATH = './result/'
if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)

"""
log 파일
"""
log = open(OUT_PATH + '/log.txt', 'w')
log.write('crawling time(UNIX):\n' + str(int(time())))
log.close()

"""
API 호출
"""
# candlestick period in seconds; valid values are 300, 900, 1800, 7200, 14400, and 86400
u = urlopen("https://poloniex.com/public?command=returnChartData&currencyPair=USDT_BTC&start=0000000000&end=9999999999&period=86400")
sleep(30) # 데이터 로딩시간

data = u.read()
j = (json.loads(data))

"""
데이터 저장
"""
for item in j:
    with open(OUT_PATH + str(item['date']) + '.json', 'w') as output:
        json.dump(item, output)
        #pprint.pprint()
