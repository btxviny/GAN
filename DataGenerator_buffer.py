
import numpy as np
import glob
import cv2
import os
import random


class DataGenerator:
    def __init__(self, path, shape, n_frames = 5, stride=6):
        self.stable_path = os.path.join(path,'stable')
        self.unstable_path = os.path.join(path,'unstable')
        self.length = n_frames
        self.stride = stride
        self.shape = shape
        self.video_names = os.listdir(self.stable_path)

    def get_paths(self,video):
        s_path = os.path.join(self.stable_path,video)
        u_path = os.path.join(self.unstable_path,video)
        paths = [s_path,u_path]
        return( paths)
    
    def __call__(self):
        h,w,c = self.shape
        self.video_names = random.sample(self.video_names, len(self.video_names))
        for video in self.video_names:
            s_path = os.path.join(self.stable_path,video)
            u_path = os.path.join(self.unstable_path,video)
            stable_cap = cv2.VideoCapture(s_path)
            unstable_cap = cv2.VideoCapture(u_path)
            # Get the frame count of the stable video
            s_frame_count = int(stable_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # Get the frame count of the unstable video
            u_frame_count = int(unstable_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            num_of_frames = min(s_frame_count,u_frame_count)
            s_buffer = np.zeros((30,h,w,3),dtype=np.float32)
            u_buffer = np.zeros((30,h,w,3),dtype=np.float32)
            for i in range(30):
                _, frame1 = stable_cap.read()
                frame1 = preprocess(frame1, self.shape)
                _, frame2 = unstable_cap.read()
                frame2 = preprocess(frame2, self.shape)
                s_buffer[i,...] = frame1
                u_buffer[i,...] = frame2
            sequence = np.zeros((h,w,self.length),dtype=np.float32)
            for frame_idx in range(30,num_of_frames):
                ret1, It_curr = unstable_cap.read()
                ret2, Igt_curr = stable_cap.read()
                if not ret1 or not ret2: break
                It = preprocess(It_curr,(h,w,c))
                Igt = preprocess(Igt_curr,(h,w,c))
                for i in range(self.length):
                    sequence[:,:,i] = cv2.cvtColor(s_buffer[-2**i,...],cv2.COLOR_BGR2GRAY)
                    
                u_buffer[:-1,...] = u_buffer[1:,...]
                u_buffer[-1,...] = It
                s_buffer[:-1,...] = s_buffer[1:,...]
                s_buffer[-1,...] = Igt

                yield sequence, It, Igt

def preprocess(img,shape):
    h,w,_ = shape
    img = cv2.resize(img,(w,h),cv2.INTER_LINEAR)
    img = (img- 127.0) / 127.0
    return img



