from detectC import *
from arcfaceC import *
from databaseM import *
from sklearn.neighbors import KNeighborsClassifier
import os
from os import listdir
from os.path import isfile, join
import os.path as osp
import glob
import cv2

import tkinter as tk
from tkinter import *
import tkinter.simpledialog
from PIL import Image, ImageTk
import imutils
import requests

def read_image(img_path, **kwargs):
    """Read and transpose input image

    Parameters
    ----------
        img_path : str
            Path to image
    Returns:
    -------
        img : numpy array[float32]
            Analyzed image
    """
    mode = kwargs.get('mode', 'rgb')
    layout = kwargs.get('layout', 'HWC')
    # Read image (transpose if necessary)
    if mode=='gray':
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    assert img is not None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if layout=='CHW':
        img = np.transpose(img, (2,0,1))
    return img

def window2center(win):
    """Centers a tkinter window
    
    Parameters
    ----------
        win : object
            Tkinter window

    """
    win.update_idletasks()
    width = win.winfo_width()
    frm_width = win.winfo_rootx() - win.winfo_x()
    win_width = width + 2 * frm_width
    height = win.winfo_height()
    titlebar_height = win.winfo_rooty() - win.winfo_y()
    win_height = height + titlebar_height + frm_width
    x = win.winfo_screenwidth() // 2 - win_width // 2
    y = win.winfo_screenheight() // 2 - win_height // 2
    win.geometry('{}x{}+{}+{}'.format(width, height, x, y))
    win.deiconify()

def EDistace(emb1,emb2):
    """Calculation of Euclidean distance of two vectors

    Parameters
    ----------
        emb1, emb2 : numpy array[float32]
            Analyzed vectors
    Returns:
    -------
        dist : float64
            Euclidean distance
    """
    dist = np.sqrt(np.sum(np.power(emb1-emb2,2)))
    return dist

def learnFromVideo(file_path, labelName, reading_step, numOfClusters):
    """Generating a face description from a video file

    Parameters
    ----------
        file_path : str
            Path to video file
        labelName : str
            Name to save
        reading_step : int
            Step of reading a video file
        numOfClusters : int
            Number of target clusters for the kmeans method

    """
    assert isinstance(reading_step, int)
    assert numOfClusters > 0

    # Face Detection
    detector = SCRFD(model_file='./models/SCRFD_10G_KPS.onnx')
    detector.exprov(-1)
    # ArcFace
    comparator = ArcFace(model_file='./models/updated_resnet100.onnx')
    comparator.exprov(-1)
    # MongoDb
    dbM = MongoDB(IP='127.0.0.1', port=27017, collection='Faces')

    # Video File Reading & Preprocessing
    cap = cv2.VideoCapture(file_path)
    face_emb = []
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # count of frames
    assert count > reading_step, "ERROR: Too short record"
    iter = 0
    while True:
        if (iter > count):
            break
        else:
            # Get frame
            cap.set(cv2.CAP_PROP_POS_FRAMES,iter)
            success, frame = cap.read()
            if success:
                # Detect face
                bboxes, kpss = detector.detect(frame, 0.5)
                if bboxes.size != 0:
                    bbox = bboxes[0]
                    # Get embedding
                    if kpss is not None:
                        kps = kpss[0]
                        ft = comparator.getEmbedding(frame, bbox, kps)
                    else:
                        ft = comparator.getEmbedding(frame, bbox)
                    ft = np.float32(ft)
                    face_emb.append(ft)
            iter = iter + reading_step
    face_emb = np.array(face_emb)
    cap.release()
    assert len(face_emb) > numOfClusters, "ERROR: Too short record"

    # Face Embeddings Clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 0.1)
    ret, label, center = cv2.kmeans(face_emb,numOfClusters,None,criteria,10,cv2.KMEANS_PP_CENTERS)
    Selection = [];
    # Selection of representatives 
    for i in range(numOfClusters):
        group = []
        for ii in range(len(face_emb)):
            if label[ii][0] == i:
                group.append(face_emb[ii])
        if len(group) <= 2:
            Selection.append(group[0].tolist())
        else:
            minDist = []
            for di in range(len(group)):
                rowDist = 0
                for dii in range(len(group)):
                    dist = EDistace(group[di],group[dii])
                    rowDist = rowDist + dist
                minDist.append(rowDist)
            index_min = np.argmin(minDist)
            Selection.append(group[index_min].tolist())
    assert len(Selection) != 0
    dbM.addItem(labelName, Selection)

def learnFromImages(folder_path, labelName):
    """Generating a face description from a image files

    Parameters
    ----------
        folder_path : str
            Path to an image files
        labelName : str
            Name to save

    """
    # Face Detection
    detector = SCRFD(model_file='./models/SCRFD_10G_KPS.onnx')
    detector.exprov(-1)
    # ArcFace
    comparator = ArcFace(model_file='./models/updated_resnet100.onnx')
    comparator.exprov(-1)
    # MongoDb
    dbM = MongoDB(IP='127.0.0.1', port=27017, collection='Faces')

    img_files = [f for f in listdir(folder_path) if isfile(join(folder_path, f))]

    ImgData = []
    for i, imgf in enumerate(img_files):
        # Load Image
        img = read_image(folder_path + "/" + imgf)
        # Detect face
        bboxes, kpss = detector.detect(img, 0.5)
        if bboxes.size != 0:
            bbox = bboxes[0]
            # Get embedding
            if kpss is not None:
                kps = kpss[0]
                ft = comparator.getEmbedding(img, bbox, kps)
            else:
                ft = comparator.getEmbedding(img, bbox)
            ft = ft.tolist()
            ImgData.append(ft)
    assert len(ImgData) != 0
    dbM.addItem(labelName, ImgData)


def getLabelDialog():
    """Label dialog 

    Return
    ------
        label : str
            Name to save

    """
    root = tk.Tk()
    root.title("Unknown person")
    window2center(root)    
    # get label
    label = tk.simpledialog.askstring(title = "Label", prompt = "Name", parent = root, initialvalue = "Unknown")
    if not label:
        label = ""
    else:
        label = label.strip()
        if not label:
            label = "Unknown"
    root.destroy() # destroy root window
    return label


def realTimeFaceDR(videoSource, url=None):
    """Real-time face detection and recognition

    Function mode : 0 - only detect faces | 1 - detect and recognize faces | 2 - learning mode
    UI Control signs : d - detection
                       r - detection and recognition
                       l - learning mode : i - recording
                                           o - stop recording and perform processing
                                           p - leave learning mode
                       e - end

    Parameters
    ----------
        videoSource : int
            Video data source indicator : 0 - binding webcam | 1 - remote camera

    """
    mode = 0
    if videoSource == 0:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        capture_size = (int(cap.get(3)), int(cap.get(4)))
    else:
        capture_size = (640,480)
    fps = 20.0
    color = (111, 255, 79) # rectangle color
    color_t = (99,255,71) # text color
    font = cv2.FONT_HERSHEY_SIMPLEX # font
    fontScale = 0.6 # fontScale
    thickness = 1 # Line thickness of 2 px
    END = False

    # Face Detection
    detector = SCRFD(model_file='./models/SCRFD_10G_KPS.onnx')
    detector.exprov(-1)
    # ArcFace
    comparator = ArcFace(model_file='./models/updated_resnet100.onnx')
    comparator.exprov(-1)
    # MongoDb
    dbM = MongoDB(IP='127.0.0.1', port=27017, collection='Faces')

    # Get Database Data
    dbIDs, dbLabels, dbEmbs = dbM.getItems()
    # kNN learning
    kNN = KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='brute', metric='cosine', metric_params=None, n_jobs=-1)
    if bool(dbEmbs):
        kNN.fit(dbEmbs, dbIDs)

    while True:
        try:
            if videoSource == 0:
                success, frame = cap.read()
            elif videoSource == 1:
                success = True
                frame_resp = requests.get(url)
                frame_arr = np.array(bytearray(frame_resp.content), dtype=np.uint8)
                frame = cv2.imdecode(frame_arr, -1)
                frame = imutils.resize(frame, width=1000, height=1800)
            if success:
                if (mode == 0) or (mode == 1):
                    # Detect faces
                    bboxes, kpss = detector.detect(frame, 0.5)
                    if bboxes.size != 0:
                        for ii in range(bboxes.shape[0]):
                            bbox = bboxes[ii]
                            # Show bounding boxes
                            x1, y1, x2, y2, score = bbox.astype(int)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            if mode == 1:
                                # Get embedding
                                if kpss is not None:
                                    kps = kpss[ii]
                                    face_em = comparator.getEmbedding(frame, bbox, kps)
                                else:
                                    face_em = comparator.getEmbedding(frame, bbox)
                                # Compare input with dataset
                                if bool(dbEmbs):
                                    prediction = kNN.predict([face_em])
                                    knh = kNN.kneighbors([face_em])
                                    class_IDs = kNN.classes_[kNN._y[knh[1][0]]]
                                    minDist = 2
                                    for iter,cid in enumerate(class_IDs):
                                        if cid == prediction[0]:
                                            minDist = min(minDist,knh[0][0][iter])
                                    # Threshold
                                    if minDist < 1:
                                        dbItem = dbM.getNameByID(prediction[0])
                                        label = dbItem['Name'] + " " + str(np.round(100-(minDist/0.02))) + "%"
                                    else:
                                        label = "Unknown"
                                else:
                                    label = "Unknown"
                                cv2.putText(frame, label, (x1+2,y1-5), font, fontScale, color_t, thickness, cv2.LINE_AA)
                elif (mode == 2):
                    LearningEND = False
                    LearningRecord = False
                    LearningFinish = False
                    # Get the face label
                    labelName = getLabelDialog()
                    if not labelName:
                        mode = 1
                        continue
                    vid_cod = cv2.VideoWriter_fourcc(*'XVID')
                    video_name = "./tests/videos/face_data_"+labelName+".mp4"
                    output = cv2.VideoWriter(video_name, vid_cod, fps, capture_size)
                    while True:
                        user_pressed = cv2.waitKey(1) & 0xFF
                        if videoSource == 0:
                            success, frame = cap.read()
                        elif videoSource == 1:
                            success = True
                            frame_resp = requests.get(url)
                            frame_arr = np.array(bytearray(frame_resp.content), dtype=np.uint8)
                            frame = cv2.imdecode(frame_arr, -1)
                            frame = imutils.resize(frame, width=1000, height=1800)
                        if success:
                            addT = "Ready"
                            if LearningRecord:
                                # Record video
                                addT = "Recording"
                                output.write(frame)
                            # UI Control
                            if (user_pressed == ord("i")):
                                addT = "Recording"
                                LearningRecord = True
                            elif user_pressed == ord("o"):
                                # Stop recording
                                addT = "Finishing"
                                if LearningRecord:
                                    LearningFinish = True
                                LearningEND = True
                            elif user_pressed == ord("p"):
                                addT = "Stop"
                                LearningFinish = False
                                LearningEND = True
                            cv2.putText(frame, "Learning -> "+addT, (2,30), font, 1, color_t, 1, cv2.LINE_AA)
                            cv2.imshow("Webcam",frame)
                        if LearningEND:
                            break
                    output.release()
                    if LearningFinish:
                        learnFromVideo(file_path=video_name, labelName=labelName, reading_step=10, numOfClusters=10)
                        # Update db Items and kNN
                        dbIDs, dbLabels, dbEmbs = dbM.getItems()
                        kNN.fit(dbEmbs, dbIDs)
                    else:
                        if os.path.exists(video_name):
                            os.remove(video_name)
                    mode = 1

                # UI Control
                if mode == 0:
                    cv2.putText(frame, 'Detection', (2,30), font, 1, color_t, 1, cv2.LINE_AA)
                elif mode == 1:
                    cv2.putText(frame, 'Recognition', (2,30), font, 1, color_t, 1, cv2.LINE_AA)
                elif mode == 2:
                    cv2.putText(frame, 'Learning', (2,30), font, 1, color_t, 1, cv2.LINE_AA)

                cv2.imshow("Webcam",frame)

                user_pressed = cv2.waitKey(1) & 0xFF
                if user_pressed == ord("d"):
                    mode = 0
                elif user_pressed == ord("r"):
                    mode = 1
                elif user_pressed == ord("l"):
                    mode = 2
                elif user_pressed == ord("e"):
                    END = True
            if END:
                break
        except Exception as e:
            print(e)
    if videoSource == 0:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    ### Image Data ###
    # learnFromImages(folder_path='./tests/images/musk', labelName="Elon")

    ### Video Data ###
    # learnFromVideo(file_path='./tests/videos/__videoFile__', labelName="NewFace", reading_step=10, numOfClusters=9)

    realTimeFaceDR(videoSource = 0)
