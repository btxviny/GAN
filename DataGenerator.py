
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
        self.frame_idx = 30

    def get_paths(self,video):
        s_path = os.path.join(self.stable_path,video)
        u_path = os.path.join(self.unstable_path,video)
        paths = [s_path,u_path]
        return( paths)
    
    def __call__(self):
        self.video_names = random.sample(self.video_names, len(self.video_names))
        for video in self.video_names:
            paths = self.get_paths(video)
            stable_frames, unstable_frames = load_video(paths,self.shape)
            n,h,w,c = stable_frames.shape
            sequence = np.zeros(shape=(h,w,self.length),dtype=np.float32)
            #It = np.zeros(shape=(h,w,c),dtype=np.float32)
            #Igt = np.zeros_like(It)
            for frame_idx in range(30,n):
                for (i,j) in zip(range(frame_idx - self.stride, frame_idx - self.length*self.stride, -self.stride) , range(self.length)):
                    sequence[:,:,j] = cv2.cvtColor(stable_frames[i,...],cv2.COLOR_BGR2GRAY)
                It = unstable_frames[frame_idx,...]
                Igt = stable_frames[frame_idx,...]
                yield sequence, It, Igt


def load_video(paths,shape):
    stable_frames = []
    unstable_frames = []
    stable_cap = cv2.VideoCapture(paths[0])
    unstable_cap = cv2.VideoCapture(paths[1])
    while True:
        ret, frame1 = stable_cap.read()
        if not ret:
            break
        frame1 = preprocess(frame1,shape)
        stable_frames.append(frame1)
        ret, frame2 = unstable_cap.read()
        if not ret:
            break
        frame2 = preprocess(frame2,shape)
        unstable_frames.append(frame2)
    stable_cap.release()
    unstable_cap.release()
    #in some video pairs the stable and unstable version dont have the same frame count
    frame_count = min(len(stable_frames),len(unstable_frames))
    stable_frames = stable_frames[:frame_count]
    unstable_frames = unstable_frames[:frame_count]
    #convert to np.arrays
    stable_frames = np.array(stable_frames,dtype=np.float32)
    unstable_frames = np.array(unstable_frames,dtype=np.float32)
    return(stable_frames,unstable_frames)

def preprocess(img,shape):
    h,w,_ = shape
    img = cv2.resize(img,(w,h),cv2.INTER_LINEAR)
    img = (img- 127.0) / 127.0
    return img

class ShuffledGenerator:
    def __init__(self,path,shape,stride,length,max_samples):
        self.stable_path = os.path.join(path,'stable')
        self.unstable_path = os.path.join(path,'unstable')
        self.video_names = os.listdir(self.stable_path)
        self.shape = shape
        self.stride = stride
        self.length = length
        self.max_samples = max_samples
    def imread(self,path,grayscale = False):
        h,w,_ = self.shape
        img = cv2.imread(path)
        if grayscale:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img,(w,h),cv2.INTER_LINEAR)
        img = (img- 127.0) / 127.0
        return img.astype(np.float32)

    def __call__(self):
        for i in range(self.max_samples):
            video = random.choice(self.video_names)
            stable_video_path = os.path.join(self.stable_path,video)
            unstable_video_path = os.path.join(self.unstable_path,video)
            total_frames = len(os.listdir(os.path.join(self.stable_path,video)))
            frame_idx = random.randint(30,total_frames - 1)
            sequence = np.zeros(self.shape[:-1] + (self.length,),dtype = np.float32)
            for i,j in zip(range(self.length),range(frame_idx - self.stride, frame_idx - self.length*self.stride, -self.stride)):
                sequence[...,i] = self.imread(os.path.join(stable_video_path,f'{j}.png'),grayscale=True)
            It = self.imread(os.path.join(unstable_video_path,f'{frame_idx}.png'))
            Igt = self.imread(os.path.join(stable_video_path,f'{frame_idx}.png'))
            yield sequence, It, Igt


