import numpy as np
import cv2
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import argparse


parser = argparse.ArgumentParser('export tracks from SORT')
parser.add_argument("--video_path", type=str, help="path to the video")
opt = parser.parse_args()

video_path=opt.video_path

index1 = max(opt.video_path.rfind('\\'), opt.video_path.rfind('/')) + 1
index2 = video_path.rfind('.')
vid_name = video_path[index1:index2]
print(vid_name)



def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    boxA[2:4] += boxA[0:2]
    boxB[2:4] += boxB[0:2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def abs_area_diff(boxA, boxB):
    boxAArea = (boxA[2] + boxA[0] + 1) * (boxA[3] + boxA[1] + 1)
    boxBArea = (boxB[2] + boxB[0] + 1) * (boxB[3] + boxB[1] + 1)
    return abs(boxAArea-boxBArea)/boxBArea




likelihood=np.load('Pixel_Probabilities/'+vid_name+'likelihood.npy')
prior=np.load('Pixel_Probabilities/'+vid_name+'prior.npy')

imm=np.zeros(prior.shape)
seq_dets = np.loadtxt('Detector_outputs\\'+vid_name+'_detections.txt', delimiter=',')  # load detections
# #p_test=(76,196)
# p_test=[76,196]
# for i in range(paths.shape[0]):
#     for j in range(paths.shape[1]):
#         #print(paths[i,j])
#         if paths[i,j]:
#             path=np.array(paths[i,j])
#             for points in path:
#                 #print(points)
#                 if p_test[0]>=points[0] and p_test[0]<=points[0]+points[2] and p_test[1]>=points[1] and p_test[1]<=points[1]+points[2]:
#                     imm[i,j]=255
#                 #print(path)
# cv2.imshow('imm',imm)
# cv2.waitKey()
cap = cv2.VideoCapture(video_path)
if (cap.isOpened() == False):
    print("Error opening video stream or file")

ret, frame = cap.read()
f = 600 / max(frame.shape)
frame = cv2.resize(frame, dsize=(0, 0), fx=f, fy=f)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
if not os.path.exists('Predictions_Output'):
    os.makedirs('Predictions_Output')
out = cv2.VideoWriter('Predictions_Output\\'+vid_name+'predictions.avi',fourcc, 20.0, (frame.shape[1], frame.shape[0]))

i=0
if (cap.isOpened() == False):
    print("Error opening video stream or file")

while(cap.isOpened()):

    ret, frame = cap.read()
    if ret==True:
        f = 600 / max(frame.shape)
        frame=cv2.resize(frame,dsize=(0,0), fx=f, fy=f)
        dets = seq_dets[seq_dets[:, 0] == i, 2:7]

        posteriori = np.zeros(prior.shape, dtype='float')

        i=i+1

        if dets is not None:
            for p in dets:

                #print(int(p[2]),int(p[3]), int(p[2]+p[4]), int(p[3]+p[5]),)
                cv2.rectangle(frame, (int(p[0]),int(p[1])), (int(p[0]+p[2]),int(p[1]+p[3])), color=(255,255,255), thickness=2)

                for a in range(likelihood.shape[0]):
                    for b in range(likelihood.shape[1]):
                        #print(paths[i,j])
                        if likelihood[a,b]:
                            path=np.array(likelihood[a,b])
                            for points in path:
                                #print(points)
                                if p[0]>=points[0]-points[2]/4 and p[0]<=points[0]+points[2]/4 and p[1]>=points[1]-points[3]/4 and p[1]<=points[1]+points[3]/4:# and \
                                        #abs_area_diff(p.copy(), points.copy())<0.5:
                                    posteriori[int(a-points[3]/2):int(a+points[3]/2),int(b-points[2]/2):int(b+points[2]/2)]=posteriori[int(a-points[3]/2):int(a+points[3]/2),int(b-points[2]/2):int(b+points[2]/2)]+1
                                    #frame[int(a-points[3]/2):int(a+points[3]/2),int(b-points[2]/2):int(b+points[2]/2)]=(frame[int(a-points[3]/2):int(a+points[3]/2),int(b-points[2]/2):int(b+points[2]/2)]/2+128).astype('uint8')
                                #print(path)
        if posteriori.max() !=0:
            posteriori=posteriori/posteriori.max()*255
            frame=frame.astype('float')
            frame[:,:,2]=frame[:,:,2]+posteriori
            frame=frame/frame.max()*255
            frame=frame.astype('uint8')
        cv2.imshow('frame', frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break
    else:
        cap.release()
        out.release()


