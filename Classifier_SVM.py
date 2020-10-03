# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 11:04:10 2020

@author: K B PRANAV
"""

from sklearn.svm import LinearSVC
import numpy as np
import cv2 
import time
import tkinter as tk
import pickle

print("Modules loaded.")
class Model:
    
    def __init__(self):
        self.model = LinearSVC(max_iter=1500)
        self.init_gui()
        
        
    def train_model(self):
        print("Loading Dataset...")
        img_list = np.array([])
        class_list = np.array([])

        for i in range(1,500):
            img = cv2.imread(f'database/train/ped/{i}.png')[:, :, 0]
            img = img.reshape(12800,)
            img_list = np.append(img_list, [img])
            class_list = np.append(class_list, 1)
            
       
        for i in range(1,500):
            img = cv2.imread(f'database/train/bkg/{i}.png')[:, :, 0]
            img = img.reshape(12800,)
            img_list = np.append(img_list, [img])
            class_list = np.append(class_list, 2)

       
        img_list = img_list.reshape(len(class_list), 12800)
        print("Training Started...")
        self.model.fit(img_list, class_list)
        print("Model successfully trained!")
        
        # save the model to disk
        filename = 'finalized_model.sav'
        pickle.dump(self.model, open(filename, 'wb'))
        
    def load_trained_model(self):
        filename = 'finalized_model.sav'
        self.model = pickle.load(open(filename, 'rb'))
        print("Trained Model loaded.")
        
    def detect(self):
        print("\n[INFO]Camera Starting....")
        cap1 = cv2.VideoCapture(0)
        time.sleep(2)
        print("\nDetecting Pedestrians now...\n[INFO]Press 'q' to exit")
        while(True):
            #name+=1
            ret1,frame1 = cap1.read()
            
            frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            img=np.array(frame)
            i=0
            l=[]
            
            while(i<8):
                crop_img = img[ :,80*i:160+80*i]
                crop_img= crop_img[:,:,0]
                crop=np.array(crop_img)
                crop=cv2.resize(crop, (80, 160))
                test_image = crop.reshape(12800,)
                result = self.model.predict([test_image])
                if(result==1):
                    l.append(i)
                
                i+=1
            for i in l:
                cv2.rectangle(frame,(80*i,10),(80*i+160,460),(0,210,0),3)
        
         
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imshow("Pedestrian Detection",frame)
            
            key = cv2.waitKey(1)&0xFF
            if(key==ord('q')):
                break
            
        cap1.release()
        cv2.destroyAllWindows()
        
        
    
    def init_gui(self):
       
        window= tk.Tk()
               
        self.btn_train = tk.Button(window ,text="Train Model", width=50, command=lambda: self.train_model())
        self.btn_train.pack(anchor=tk.CENTER, expand=True)
        
        self.btn_load = tk.Button(window,text="Load Trained Model", width=50, command=lambda: self.load_trained_model())
        self.btn_load.pack(anchor=tk.CENTER, expand=True)
        
        
        self.btn_detect = tk.Button(window,text="Detect Pedestrian", width=50, command=lambda: self.detect())
        self.btn_detect.pack(anchor=tk.CENTER, expand=True)
        
        window.mainloop()
        
Model()
