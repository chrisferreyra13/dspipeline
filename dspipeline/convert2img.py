import cv2
import imutils
import os
from imutils.video import VideoStream
from skimage import data, filters
import numpy as np
import math

# Convertir en pipeline!
def convert_video_to_image(video_folder, video_file):
    # video_folder='/home/administrator/detectron2/videos'
    # video_file = "cam1-20210414092200.mp4"
    video_name,_ = os.path.splitext(video_file)
    video_path = os.path.join(video_folder,video_file)


    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        exit(0)

    frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)

    # Store selected frames in an array
    frames = []
    for fid in frameIds:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    # Calculate the median along the time axis
    medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)   
    # Convert background to grayscale
    grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)

    grayMedianFrame  =cv2.GaussianBlur(grayMedianFrame, (21,21),0)
    # cv2.startWindowThread()
    # cv2.namedWindow("frame",cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("frame", (640, 480))

    ##------------------

    OPENCV_OBJECT_TRACKERS = {
    "MOG": cv2.createBackgroundSubtractorMOG2,
    "KNN": cv2.createBackgroundSubtractorKNN
        }

    # cv2.namedWindow("background",cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("background", ( 640,480 ) )

    # cv2.namedWindow("foreground",cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("foreground", ( 640,480 ) )
    bgsubstractor=OPENCV_OBJECT_TRACKERS["KNN"]()
    ##------------------
    def RemoveBackGround(bgsubstractor, frame):
        #frame=imutils.resize(frame, height=480)
        background=bgsubstractor.apply(frame)
        #background=bgsubstractor.apply(frame)

        # cv2.imshow("background", background)

        foreground=cv2.bitwise_and(frame, frame, mask=background)
        # cv2.imshow("foreground", foreground)
        return foreground,background


    # Reset frame number to 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    os.mkdir(f'/home/administrator/detectron2/videos_processed/{video_name}')
    no_movement = f'/home/administrator/detectron2/videos_processed/{video_name}/no_movement'
    movement = f'/home/administrator/detectron2/videos_processed/{video_name}/movement'
    os.mkdir(no_movement)
    os.mkdir(movement)
    frame_counter=0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert current frame to grayscale
        grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        grayframe  =cv2.GaussianBlur(grayframe, (21,21),0)

        # Calculate absolute difference of current frame and 
        # the median frame
        dframe = cv2.absdiff(grayframe, grayMedianFrame)
        # Treshold to binarize
        th, dframe = cv2.threshold(dframe, 50, 255, cv2.THRESH_BINARY)
        # Display the resulting frame
        OKcount1 = np.sum(dframe > 0)/dframe.size
        cv2.putText(dframe,str(round(OKcount1,2)), (50,100), cv2.FONT_HERSHEY_SIMPLEX, 4, (200,200,100))
        # cv2.imshow('frame',dframe)
        # fore,back = RemoveBackGround(bgsubstractor, frame)
        
        # fore  =cv2.GaussianBlur(fore, (51,51),0)
        # OKcount2 = np.sum(fore > 0)/fore.size
        # cv2.putText(fore,str(round(OKcount2,2)), (50,100), cv2.FONT_HERSHEY_SIMPLEX, 4, (200,200,100))    
        # cv2.imshow('foreground',fore)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        if OKcount1>.05:
            save_file = os.path.join(movement, f"{video_name}_{frame_counter}.jpg")
        else:
            save_file = os.path.join(no_movement, f"{video_name}_{frame_counter}.jpg")
        cv2.imwrite(save_file, frame)
        frame_counter+=1

    # When everything done, release the capture
    # cap.release()
    # cv2.destroyAllWindows()