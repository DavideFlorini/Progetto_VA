import numpy as np
import cv2
import argparse
import os


parser = argparse.ArgumentParser('export tracks from SORT')
parser.add_argument("--video_path", type=str, help="path to the video")

opt = parser.parse_args()

video_path=opt.video_path


index1 = max(opt.video_path.rfind('\\'), opt.video_path.rfind('/')) + 1
index2 = video_path.rfind('.')

vid_name = video_path[index1:index2]
print(vid_name)

track = np.loadtxt('Tracker_outputs/'+vid_name+'_track.txt',delimiter=',') #load detections
track=np.vstack(track)
track_end=int(track[-1,0])
cap = cv2.VideoCapture(video_path)
if (cap.isOpened() == False):
    print("Error opening video stream or file")

ret, frame = cap.read()
f = 600 / max(frame.shape)
frame = cv2.resize(frame, dsize=(0, 0), fx=f, fy=f)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
cap.release()

prob=np.zeros([frame.shape[0], frame.shape[1]], dtype='float')
tracks=np.empty([1,2],dtype=object)
likelihood=np.empty(prob.shape, dtype=object)
for i in range(likelihood.shape[0]):
    for j in range(likelihood.shape[1]):
        likelihood[i,j]=[]
i=0
while(i<track_end):
    if True==True:

        i=i+1
        print(i)
        points=track[track[:,0]==i]
        points = points.astype('int')
        #comb = frame.copy()

        if points is not None:
            #if np.random.binomial(1, 0.2) == 1:
                for p in points:
                    prob[p[3]:p[3]+p[5], p[2]:p[2]+p[4]]=prob[p[3]:p[3]+p[5],p[2]:p[2]+p[4]]+np.ones(prob[p[3]:p[3]+p[5],p[2]:p[2]+p[4]].shape)
                    if tracks[0,0] is not None:
                        a=tracks[:, 0]
                        itemindex = np.nonzero(tracks[:, 0] == p[1])
                        if np.any(tracks[:,0] == p[1]):
                            tracks[itemindex,1][0][0].append(list((p[2], p[3], p[4], p[5])))
                        else:
                            tracks=np.vstack((tracks, np.array([p[1], list([])])))
                            tracks[-1,1].append(list((p[2], p[3], p[4], p[5])))
                    else:
                            tracks[0, 0]=p[1]
                            tracks[0, 1]=[]
                            tracks[0,1].append(list((p[2], p[3], p[4], p[5])))

                    itemindex = np.nonzero(tracks[:, 0] == p[1])
                    for point_track in np.array(tracks[itemindex,1][0][0]):

                        if len(likelihood[int(p[3] + p[5] / 2), int(p[2] + p[4] / 2)])<2000:

                            likelihood[int(p[3]+p[5]/2), int(p[2]+p[4]/2)].append(list(point_track))

        points_pre=points

        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #  break
if not os.path.exists('Pixel_Probabilities'):
    os.makedirs('Pixel_Probabilities')

np.save('Pixel_Probabilities/'+vid_name+'_likelihood.npy', likelihood)
np.save('Pixel_Probabilities/'+vid_name+'_prior.npy', prob)
