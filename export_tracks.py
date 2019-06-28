import cv2
from pygments.lexer import default

from sort import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse


parser = argparse.ArgumentParser('export tracks from SORT')
parser.add_argument("--video_path", type=str, help="path to the video")
parser.add_argument("--Display", type=bool, default=False, help="Display")

opt = parser.parse_args()

video_path=opt.video_path

index1 = max(opt.video_path.rfind('\\'), opt.video_path.rfind('/')) + 1
index2 = video_path.rfind('.')

vid_name = video_path[index1:index2]
print(vid_name)


mot_tracker = Sort()
display=opt.Display


total_time = 0.0
total_frames = 0
colours = np.random.rand(32, 3)  # used only for display

if (display):
    cap = cv2.VideoCapture(video_path)
    if (cap.isOpened() == False):
        print("Error opening video stream or file")
    plt.ion()
    fig = plt.figure()


seq_dets = np.loadtxt('Detector_outputs\\'+vid_name+'_detections.txt', delimiter=',')  # load detections
if not os.path.exists('Tracker_outputs'):
    os.makedirs('Tracker_outputs')

mot_tracker = Sort() #create instance of the SORT tracker

with open('Tracker_outputs\\'+vid_name+'_track.txt', 'w') as out_file:
    for frame in range(int(seq_dets[:, 0].max())):
        frame += 1  # detection and frame numbers begin at 1
        dets = seq_dets[seq_dets[:, 0] == frame, 2:7]
        dets[:, 2:4] += dets[:, 0:2]  # convert to [x1,y1,w,h] to [x1,y1,x2,y2]
        total_frames += 1

        if (display):
            ax1 = fig.add_subplot(111, aspect='equal')
            str_i = str(frame).zfill(3)
            ret, im = cap.read()
            ax1.imshow(im)
            plt.title(' Tracked Targets')

        start_time = time.time()
        trackers = mot_tracker.update(dets)
        cycle_time = time.time() - start_time
        total_time += cycle_time

        for d in trackers:
            #print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1,%.2f,%.2f' % (frame, d[4], d[0], d[1], d[2] - d[0], d[3] - d[1], d[5], d[6]),
                  #file=out_file)
            print('%d,%d,%.3f,%.3f,%.3f,%.3f' % (frame, d[4], d[0], d[1], d[2] - d[0], d[3] - d[1]),
                  file=out_file)
            #print(d)
            if (display):
                d = d.astype(np.int32)
                ax1.add_patch(patches.Rectangle((d[0], d[1]), d[2] - d[0], d[3] - d[1], fill=False, lw=3,
                                                ec=colours[d[4] % 32, :]))
                ax1.set_adjustable('box-forced')
                ax1.arrow(d[0] + (d[2]-d[0])/2, d[1] + (d[3]-d[1])/2, d[5]*10, d[6]*10, head_width=20)

        if (display):
            fig.canvas.flush_events()
            plt.draw()
            ax1.cla()

print("Total Tracking took: %.3f for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))
if (display):
    print("Note: to get real runtime results run without the option: --display")
    cap.release()
