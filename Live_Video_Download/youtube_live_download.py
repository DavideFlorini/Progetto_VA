import os
import argparse
import funzioni
import csv


parser = argparse.ArgumentParser(description="Download streaming webcam")
parser.add_argument("link", type=str, help="link")
parser.add_argument("t", type=int, help="time (s)")

args = parser.parse_args()

url=args.link
tempo=args.t

flag = 0

#print(f'link: {url} ')

#try:
#    ret=funzioni.trova_youtube_dl(url)
#    if ret!=-1:
#        [video_url, title]=ret
#        flag=1
#except:
ret=funzioni.trova_m3u8(url)
if ret!=-1:
    [video_url, title]=ret
    flag=1

if flag==1:
    if not os.path.exists('Downloaded_videos'):
        os.makedirs('Downloaded_videos')
    print('m3u8 found: ', video_url)
    command = 'ffmpeg -i ' + video_url + ' -c copy -t ' + str(tempo) + ' ' + 'Downloaded_videos\\' + title + '.mp4'
    print('command: ', command)
    os.system(command)

else:
    print('video not available: ',url)