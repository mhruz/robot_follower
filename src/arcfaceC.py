import cv2
import numpy as np
import onnx
import onnxruntime
from skimage import transform as trans
import sys
import os
import os.path as osp
from scipy import misc
import sklearn
from sklearn.decomposition import PCA
from time import sleep
from easydict import EasyDict as edict

class ArcFace:
    """
    ArcFace is a CNN based model for face recognition

    Attributes
    ----------
        model_file : str
            Path to model
        session : object, optional
            Onnxruntime inference session
    Methods
    -------
        exprov(ctx_id)
            Set session provider
        preprocess(img, bbox=None, landmark=None)
            To generate the aligned face image is input image preprocessed by affine transformation
        getEmbedding(img, bbox, kps=None)
            Generating embedding from image data
    """ 
    def __init__(self, model_file, session=None):
        """
        Parameters
        ----------
            model_file : str
            
            session : object, optional
           
        """
        self.model_file = model_file
        self.image_size = (112, 112)
        self.session = session
        if self.session is None:
            assert self.model_file is not None
            assert osp.exists(self.model_file)
            self.session = onnxruntime.InferenceSession(model_file, None)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def exprov(self, ctx_id):
        """Set session provider

        Parameters
        ----------
            ctx_id : int
                Provider indicator
            
        """
        if ctx_id < 0:
            self.session.set_providers(['CPUExecutionProvider'])
        else:
            self.session.set_providers(['CUDAExecutionProvider'])

    def preprocess(self, img, bbox=None, landmark=None, **kwargs):
        """To generate the aligned face image is input image preprocessed by affine transformation

        Parameters
        ----------
            img : numpy array[uint8]
                Analyzed image                
            bbox : numpy array[float32], optional
                Bounding box (x1, y1, x2, y2, score)
            landmark : numpy array[float32], optional
                Key points

        Returns:
        -------
            warped : numpy array[float32]
                Aligned face image
            ret : numpy array[float32]
                Aligned face image
        """
        M = None   
        # Do alignment using landmark points
        if landmark is not None:
            assert len(self.image_size)==2
            src = np.array([
              [30.2946, 51.6963],
              [65.5318, 51.5014],
              [48.0252, 71.7366],
              [33.5493, 92.3655],
              [62.7299, 92.2041] ], dtype=np.float32 )
            if self.image_size[1]==112:
                src[:,0] += 8.0
            #dst = landmark.astype(np.float32)
            tform = trans.SimilarityTransform()
            tform.estimate(landmark, src)
            M = tform.params[0:2,:]
            assert len(self.image_size)==2
            warped = cv2.warpAffine(img,M,(self.image_size[1],self.image_size[0]), borderValue = 0.0)
            return warped
    
        # If no landmark points available, do alignment using bounding box. If no bounding box available use center crop
        if M is None:
            if bbox is None:
                det = np.zeros(4, dtype=np.int32)
                det[0] = int(img.shape[1]*0.0625)
                det[1] = int(img.shape[0]*0.0625)
                det[2] = img.shape[1] - det[0]
                det[3] = img.shape[0] - det[1]
            else:
                det = bbox
            margin = kwargs.get('margin', 44)
            bb = np.zeros(4, dtype=np.int32)
            bb[0] = np.maximum(det[0]-margin/2, 0)
            bb[1] = np.maximum(det[1]-margin/2, 0)
            bb[2] = np.minimum(det[2]+margin/2, img.shape[1])
            bb[3] = np.minimum(det[3]+margin/2, img.shape[0])
            ret = img[bb[1]:bb[3],bb[0]:bb[2],:]
            if len(self.image_size)>0:
                ret = cv2.resize(ret, (self.image_size[1], self.image_size[0]))
            return ret

    def getEmbedding(self, img, bbox, kps=None):
        """Generating embedding from image data

        Parameters
        ----------
            img : numpy array[uint8]
                Analyzed image
            bbox : numpy array[float32]
                Bounding box (x1, y1, x2, y2, score)
            kps : numpy array[float32], optional
                Key points
        Returns:
        -------
            embeddings : numpy array[float32]
                Descriptive array of numbers (size: 512)
        """
        if kps is not None:
            nimg = self.preprocess(img, bbox=bbox, landmark=kps)
        else:
            nimg = self.preprocess(img, bbox, landmark=None)
        nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
        aligned_img = np.transpose(nimg, (2,0,1))
        input_blob = np.expand_dims(aligned_img, axis=0)
        input_blob = input_blob.astype(np.float32)
        embedding = self.session.run([self.output_name], {self.input_name: input_blob})
        # Postprocess
        embedding = sklearn.preprocessing.normalize(embedding[0]).flatten()
        return embedding
