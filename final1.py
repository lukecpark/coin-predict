#import module
import requests
import urllib.request
import csv
import pymysql
import math
from konlpy.tag import *
from bs4 import BeautifulSoup

#define
con=pymysql.connect(host='localhost',port=3306,user='root',password='gee9494',db='world',charset='utf8')
cursor=con.cursor()

index=1
dic={'01':'Jan','02':'Feb','03':'Mar','04':'Apr','05':'May','06':'Jun','07':'Jul','08':'Aug','09':'Sep','10':'Oct','11':'Nov','12':'Dec'}
date={'01':'1','02':'2','03':'3','04':'4','05':'5','06':'6','07':'7','08':'8','09':'9'}
year_dic={'2017':'17','2016':'16','2015':'15','2014':'14','2013':'13','2012':'12','2011':'11','2010':'10','2009':'09','2008':'08','2007':'07','2006':'06','2005':'05','2004':'04'}
firm_code={'samsung':'%EC%82%BC%EC%84%B1%EC%A0%84%EC%9E%90','lg':'lg%EC%A0%84%EC%9E%90','hyundai':'%C7%F6%B4%EB%C0%DA%B5%BF%C2%F7','sk':'sk%C7%CF%C0%CC%B4%D0%BD%BA'}
fin_code1={'samsung':'151610035517112'}
fin_code2={'samsung':'kVKOWfmoD4uc0ATnppGAAw'}
his=[]
count_news={}
num_news=0
addr_format1='https://search.naver.com/search.naver?where=news&query='
addr_format2='&ie=utf8&sm=tab_srt&sort=1&photo=0&field=1&reporter_article=&pd=3&ds=1960.01.01&de='
addr_format3='&docid=&nso=so%3Add%2Cp%3Afrom19600101to'
addr_format4='%2Ca%3At&mynews=0&mson=0&refresh_start=0&related=0'
last_date=""
fin_format1='http://www.google.com/finance/historical?cid='
fin_format2='&startdate='
fin_format3='&enddate=May+21%2C+2017&num=30&ei='
fin_format4='&output=csv'

#function
def crawler(addr):
    global index
    global last_date
    global num_news
    if(index>160):
        return
    r=requests.get(addr+str(index))
    index+=1
    soup=BeautifulSoup(r.content,"html.parser")
    results=soup.find_all('a',{"class":"tit"})
    
    if(len(results)==0):
        return 
    for text in results:
        time=text.find_next("span",{"class":"time"})
        time_text=time.text
        if(index==161):
            last_date=time.text
        ls=time_text.split('.')
        if(text.text not in his):
            if(len(ls)==3):
                if(int(ls[2])<10):
                    day=date[ls[2]]
                else:
                    day=ls[2]
                trans_date=day+'-'+dic[str(ls[1])]+'-'+year_dic[str(ls[0])]
                if(trans_date not in count_news.keys()):
                    count_news[trans_date]=1
                else:
                    count_news[trans_date]+=1
                writer.writerow([trans_date,text.text])
            else:
                if(time_text not in count_news.keys()):
                    count_news[time_text]=1
                else:
                    count_news[time_text]+=1
                writer.writerow([time_text,text.text])
            num_news+=1
                
            his.append(text.text)
    crawler(addr)

#section1
firm_name=input('Firm name : ')
year=int(input('Until? : '))
year-=1
fw=open('news_'+firm_name+'.csv','w',encoding='utf-8',newline='')
writer=csv.writer(fw,delimiter=',')
writer.writerow(['time','title'])
addr=addr_format1+firm_code[firm_name]+addr_format2+'2017.05.21'+addr_format3+'20170521'+addr_format4
fin_addr=fin_format1+fin_code1[firm_name]+fin_format2+'Jan+1%2C'+str(year)+fin_format3+fin_code2[firm_name]+fin_format4
fname,header=urllib.request.urlretrieve(fin_addr,firm_name+'.csv')
crawler(addr)
fw.close()

#section2
page=0
while(True):
    page+=1
    index=0
    date_list=last_date.split('.')
    print(date_list[0])
    print('['+str(page)+']'+last_date)
    if(int(date_list[0])==year):
        break
    fw=open('news_'+firm_name+'.csv','a',encoding='utf-8',newline='')
    writer=csv.writer(fw,delimiter=',')
    addr=addr_format1+firm_code[firm_name]+addr_format2+date_list[0]+date_list[1]+date_list[2]+addr_format3+date_list[0]+'-'+date_list[1]+'-'+date_list[2]+addr_format4
    crawler(addr)
    fw.close()
    
#import table
cursor.execute("create table news_"+firm_name+" ( time varchar(50), title varchar(200))")
con.commit()
fr=open('news_'+firm_name+'.csv','r',encoding='utf-8',newline='')
csvreader=csv.reader(fr)
for time,title in csvreader:
    if(time=="time"):
       continue
    cursor.execute("""insert into news_"""+firm_name+"""(time,title) values (%s,%s)""",(time,title))
fr.close
con.commit()

#join table
cursor.execute("select s.Date,n.title,((s.close-s.open)/s.open*100) as percent from news_"+firm_name+" n, "+firm_name+" s where s.Date=n.time")

#text_mining
print("Text mining execution...")
komoran=Komoran()
ls1=[]
dic1={}
dic2={}
bow={}
word_in={}
doc={}

for (time,title,percent) in cursor.fetchall():
    elements=komoran.nouns(title)
    for i in elements:
        element=i.replace(' ','')
        if(element not in word_in.keys()):
            if(element not in doc.keys()):
                doc[element]=[]
                doc[element].append(title)
                word_in[element]=1
            elif(title not in doc[element]):
                word_in[element]=1
                doc[element].append(title)
        else:
            if(element not in doc.keys()):
                doc[element]=[]
                doc[element].append(title)
                word_in[element]+=1
            elif(title not in doc[element]):
                word_in[element]+=1
                doc[element].append(title)
        if(element not in ls1):
            if(abs(float(percent))>=0.75):
                if(float(percent)>0):
                    dic1[element]=1
                else:
                    dic1[element]=0
                dic2[element]=1
                ls1.append(element)
        else:
            if(abs(float(percent))>=0.75):
                if(float(percent)>0):
                    dic1[element]+=1
                dic2[element]+=1

key_list=list(dic1.keys())
key_list.sort()
fw=open("word_bias_"+firm_name+".csv","w",encoding='utf-8',newline='')
csvwriter=csv.writer(fw,delimiter=',')
csvwriter.writerow(['word','bias','df'])
for k in key_list:
        if(dic2[k]>=10):
            csvwriter.writerow([k,dic1[k]/dic2[k],float(word_in[k])/num_news])
            if(float(word_in[k])/num_news<0.1):
                bow[k]=dic1[k]/dic2[k]
fw.close()

#determine
print("Determining...")
idx={}
pos={}
cursor.execute("select * from news_"+firm_name)
fw=open("title_"+firm_name+".csv","w",encoding='utf-8',newline='')
csvwriter=csv.writer(fw,delimiter=',')
csvwriter.writerow(['time','title','idx'])
for (time,title) in cursor.fetchall():
    total=0.0
    cnt=0
    for ky in bow.keys():
        if(title.find(ky)!=-1):
            total+=float(bow[ky])
            cnt+=1
    if(cnt==0):
        if(time not in idx.keys()):
            idx[time]=0.5
        else:
            idx[time]+=0.5
    else:    
        if(time not in idx.keys()):
            idx[time]=total/cnt
        else:
            idx[time]+=total/cnt
    if(cnt!=0):
        csvwriter.writerow([time,title,total/cnt])
fw.close()

cursor.execute("select * from news_"+firm_name+" group by time")
for (time,title) in cursor.fetchall():
    if(time not in idx.keys()):
        pos[time]=0.0
    else:
        pos[time]=float(idx[time])/count_news[time]


#table join(ex_rate - firm - pos)
cursor.execute("select s.date, s.volume,((e.close-e.open)/e.open*100) as exrate, ((k.close-k.open)/k.open*100) as kospi,((s.close-s.open)/s.open*100) as percent from "+firm_name+" s, news_"+firm_name+" n, exrate e, kospi k where s.Date=n.time and n.time=e.date and e.date=k.date group by s.date")
fw=open("final_"+firm_name+".csv","w",encoding='utf-8',newline='')
csvwriter=csv.writer(fw,delimiter=',')
csvwriter.writerow(['date','volume','exrate','kospi','pos','issue','percent'])
for (time,volume,exrate,kospi,percent) in cursor.fetchall():
    if(time in pos.keys()):
        if(percent>0):
            csvwriter.writerow([time,volume,exrate,kospi,pos[time],float(count_news[time])/num_news,1])
        else:
            csvwriter.writerow([time,volume,exrate,kospi,pos[time],float(count_news[time])/num_news,-1])

fw.close()
              
#end process
cursor.execute("drop table news_"+firm_name)
con.commit()    
con.close()
print("------------------------------")
print("news_"+firm_name+".csv")
print("word_bias_"+firm_name+".csv")
print("title_"+firm_name+".csv")
print("final_"+firm_name+".csv")
print("------------------------------")
print("All process complete!")
