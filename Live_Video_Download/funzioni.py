import requests
from bs4 import BeautifulSoup
import re
import time
import youtube_dl
import string
import random
import csv
import datetime
import subprocess


def trova_m3u8( url ):
    try:
        video_url = subprocess.check_output('youtube-dl -g ' + url, shell=True)
        video_url = str(video_url[:-1], 'utf-8')
        title = subprocess.check_output('youtube-dl -e ' + url, shell=True)
        title = str(title[:-1], 'utf-8')
        title=clean_title(title)
        print(title, video_url)

        #video_url = video['url']
        #title=video['title']
        return [video_url, title]
    except:
        return -1


def trova_m3u8_mio( url ):
    req = requests.get(url)
    soup = BeautifulSoup(req.text, 'lxml')

    text = str(soup)
    pos = text.find('m3u8')
    if (pos == -1):
        #raise ValueError('m3u8 link not found')
        return -1
    else:
        starts = [match.start() for match in re.finditer(re.escape('m3u8'), text)]
        for i in starts:
            m1_s = text.rfind('\'', 0, pos) + 1
            m2_s = text.find('\'', pos)
            m1_d = text.rfind('\"', 0, pos) + 1
            m2_d = text.find('\"', pos)
            link = text[max(m1_s, m1_d):min(m2_s, m2_d)]
            #print(link)
            if link.find('http') != -1:
                flag = 1
                #print('break')
                break
        video_url = link.replace('\\', '')

        title = str(soup.title.string)
        localtime = time.asctime(time.localtime(time.time()))
        title=title+'__'+localtime

        title=clean_title(title)
        return [video_url, title]


def trova_titolo( url ):
    req = requests.get(url)
    soup = BeautifulSoup(req.text, 'lxml')

    text = str(soup)

    title = str(soup.title.string)
    #title=title_ingore(title)
    return title[:-10]


def title_ingore(title):
    title = title.encode('ascii', 'ignore')
    title=str(title, 'utf-8')
    return title


def clean_title(title):
    title = title.replace(':', '_')
    title = title.replace(' ', '_')
    title = title.replace('\\', '_')
    title = title.replace('/', '_')
    title = title.replace('|', '_')
    #title = title.decode('utf-8', 'ignore').encode("utf-8")
    title = title.encode('ascii', 'ignore')
    title=str(title, 'utf-8')
    return title

def trova_youtube_dl( url ):
    try:
        ydl = youtube_dl.YoutubeDL({'outtmpl': '%(id)s%(ext)s'})
        with ydl:
            result = ydl.extract_info(url, download=False) # We just want to extract the info)

        if 'entries' in result:
            # Can be a playlist or a list of videos
            video = result['entries'][0]
        else:
            # Just a video
            video = result
        #print(video['description'])
        video_url = video['url']
        #print('url found: ', video_url)
        try:
            title=video['title']
        except:
            title='aaa'

        title=clean_title(title)

        return [video_url, title]

    except:
        return -1


def get_idx_shuffle(p,size, list1, list2):
    num_day=int(p/100*size)
    num_night=int((100-p)/100*size)
    list1_len = list1.__len__()
    list2_len = list2.__len__()
    list1_idx = list(range(0, list1_len))
    list2_idx = list(range(0, list2_len))

    random.shuffle(list1_idx)
    random.shuffle(list2_idx)
    #print('p = ', num_day/(num_night+num_day), 'tot = ', num_day+num_night)
    return list1_idx[0:num_day], list2_idx[0:num_night]



def night_day_list(csv_file, day_threshold, night_threshold):

    now_UTC = datetime.datetime.now(datetime.timezone.utc)
    night_list=[]
    day_list=[]

    with open(csv_file, encoding='utf-8') as links:
        csv_reader = csv.reader(links, delimiter=';')
        for idx, line in enumerate(csv_reader):
            if idx != 0:
                GMT_Offset = line[3]
                local_time = now_UTC + datetime.timedelta(hours=int(GMT_Offset))
                # print(idx, line[0], line[3], local_time.hour)
                if local_time.hour >= day_threshold and local_time.hour < night_threshold:
                    line.append(local_time.hour)
                    day_list.append(line)
                    # print(line)
                else:
                    line.append(local_time.hour)
                    night_list.append(line)
                    # print(line)

    return day_list, night_list


def returnLinkList(p, num, day_th, night_th, csv_file):
    day_list, night_list = night_day_list(csv_file, day_th, night_th)
    day_idx, night_idx = get_idx_shuffle(p, num, day_list, night_list)

    list1=[day_list[i][0] for i in day_idx]
    list1=list1+[night_list[i][0] for i in night_idx]

    return list1
