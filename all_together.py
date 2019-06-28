import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--video_path", type=str, help="path to the video")
parser.add_argument("--num_frame", type=int, default=-1 , help="number of frame analyzed")

opt = parser.parse_args()


command = 'python get_detections.py --video_path ' + opt.video_path + ' --num_frame ' + str(opt.num_frame)
os.system(command)

command = 'python export_tracks.py --video_path ' + opt.video_path
os.system(command)

command = 'python Pixel_probabilities.py --video_path ' + opt.video_path
os.system(command)

command = 'python predict_video.py --video_path ' + opt.video_path
os.system(command)

exit()