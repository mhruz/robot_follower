# Face Detection & Recognition #

## Overview ##
This project represents the implementation and example of the use of state-of-the-art modules for face detection and its subsequent recognition within the stored dataset. The labelled face dataset is stored in a Mongo database. Each entry in this database has 3 attributes: _id, Name (face label), Emb (list of embeddings). 

## Prerequisites ##
The following packages need to be installed before usage:
* Os-sys - pip install os-sys
* Glob2 - pip install glob2
* Onnx - pip install onnx (1.10.2 was tested)
* Onnxruntime - pip install onnxruntime (1.10.0 was tested)
* Numpy - pip install numpy (1.21.5 was tested)
* OpenCV - pip install opencv-python (4.5.3.56 was tested)
* Scikit-learn - pip install -U scikit-learn (1.0.2 was tested)
* Scikit-image - pip install scikit-image (0.18.3 was tested)
* EasyDict - pip install easydict (1.9 was tested)
* Imutils - pip install imutils (0.5.4 was tested)
* Requests - pip install requests (2.27.1 was tested)
* Pillow - pip install Pillow (8.4.0 was tested)
* Pymongo - pip install pymongo

Next you need to download the arcface model ([see below](#arcface_note)).

## Interface ##
Use the file *src/FaceDaR.py* to run the tool. Thie file contains basic methods for user interaction and image data processing. The main method of this tool is *realTimeFaceDR (...)*, which can connect to the set camera and process user keyboard commands. <br />

**UI Control signs** 
* d - detection mode
* r - detection and recognition mode
* l - learning mode
  * i - recording
  * o - stop recording and perform processing
  * p - leave learning mode without processing
* e - end

## Used approaches ##
> ### Sample and Computation Redistribution for Efficient Face Detection ###
> SCRFD is an efficient high accuracy face detection approach <br />
> <br />
> Source: https://github.com/deepinsight/insightface/tree/master/detection/scrfd
<br />

> ### ArcFace ###
>ArcFace is a CNN based model for face recognition which learns discriminative features of faces and produces embeddings for input face images. To enhance the discriminative power of softmax loss, a novel supervisor signal called additive angular margin (ArcFace) is used here as an additive term in the softmax loss. For each face image, the model produces a fixed length embedding vector corresponding to the face in the image.<br />
> <br />
> Source: https://github.com/onnx/models/tree/master/vision/body_analysis/arcface<br />
> <br />
>> #### !!! NOTE !!! <a name="arcface_note"></a> ####
>> Due to its memory requirements, the arcface model is not part of the repository. The relevant onnx file is available at the link below. After downloading it, put it in the models directory.<br />
>> <br />
>> https://www.dropbox.com/s/co0b6otejhu6qgb/updated_resnet100.onnx?dl=0
