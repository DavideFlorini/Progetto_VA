import os
from multiprocessing.dummy import Pool as ThreadPool
import sys

sys.path.insert(0, 'Live_Video_Download')

import funzioni

def function(link):
    time=20
    command = 'python Live_Video_Download\\youtube_live_download.py ' + link + ' ' + str(time)
    os.system(command)


number=10
percentage_day=50
day_threshold=8
night_threshold=19
cam_list=funzioni.returnLinkList(p=50, num=20, day_th=day_threshold, night_th=night_threshold, csv_file='youtube (version titolo).csv')
print(cam_list)


pool = ThreadPool(3)
results = pool.map(function, cam_list)

pool.close()
pool.join()