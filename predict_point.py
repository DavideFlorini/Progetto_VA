import cv2
import numpy as np
import argparse



parser = argparse.ArgumentParser('export tracks from SORT')
parser.add_argument("--video_path", type=str, help="path to the video")
opt = parser.parse_args()

video_path=opt.video_path

index1 = max(opt.video_path.rfind('\\'), opt.video_path.rfind('/')) + 1
index2 = video_path.rfind('.')
vid_name = video_path[index1:index2]
print(vid_name)

cap = cv2.VideoCapture(video_path)
if (cap.isOpened() == False):
    print("Error opening video stream or file")

ret, frame = cap.read()
f = 600 / max(frame.shape)
frame = cv2.resize(frame, dsize=(0, 0), fx=f, fy=f)
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
cap.release()

# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
refPt = []
cropping = False


likelihood=np.load('Pixel_Probabilities/'+vid_name+'likelihood.npy')
prior=np.load('Pixel_Probabilities/'+vid_name+'prior.npy')
#cv2.imshow('prior', prior.astype('uint8'))
def click(event, x, y, flags, param):
    # grab references to the global variables
    global refPt

    # if the left mouse button was clicked, record the
    # (x, y) coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [x, y]




# load the image, clone it, and setup the mouse callback function
#image = cv2.imread(args["image"])
cv2.namedWindow("frame")
cv2.setMouseCallback("frame", click)
print('enter "q" to quit')
print('click on the image "frame" to get the predicted path for the selected point')
while True:
    # display the image and wait for a keypress
    cv2.imshow("frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if refPt:
        print(refPt)
        p=[refPt[0], refPt[1]]
        posteriori=np.zeros((frame.shape[0],frame.shape[1]))
        for a in range(likelihood.shape[0]):
            for b in range(likelihood.shape[1]):
                #print(likelihood[i,j])
                if likelihood[a,b]:
                    path=np.array(likelihood[a,b])
                    for points in path:
                        #print(points)
                        if p[0]>=points[0] and p[0]<=points[0]+points[2] and p[1]>=points[1] and p[1]<=points[1]+points[3]:
                            #cv2.rectangle(frame, (int(points[0]),int(points[1])), (int(points[0]+points[2]),int(points[1]+points[3])), color=(255,255,255), thickness=2)
                            posteriori[int(a-points[3]/2):int(a+points[3]/2),int(b-points[2]/2):int(b+points[2]/2)]=posteriori[int(a-points[3]/2):int(a+points[3]/2),int(b-points[2]/2):int(b+points[2]/2)]+1/len(path)

        if posteriori.max() !=0:
            posteriori = posteriori / prior[p[1], p[0]]
            posteriori=np.multiply(posteriori, prior)
            posteriori=(posteriori-posteriori.min()+1)/(posteriori.max()-posteriori.min()+1)
            posteriori[posteriori!=0]=np.log(posteriori[posteriori!=0])
            posteriori=(posteriori-posteriori.min())/(posteriori.max()-posteriori.min())*255
            image=frame.copy().astype('float')
            image[:,:,2]=image[:,:,2]+posteriori
            image=image/image.max()*255
            image=image.astype('uint8')
            posteriori=posteriori.astype('uint8')
            cv2.imshow('image', image)
            cv2.imshow('lik', posteriori)
        else:
            print('no path found')
        refPt.clear()
    if key == ord("q"):
        break

# close all open windows
cv2.destroyAllWindows()