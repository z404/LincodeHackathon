import os
from os import listdir
from os.path import isfile, join
import cv2
from matplotlib import pyplot as plt
import numpy as np


ImagePath = r"C:/Users/chira/OneDrive/Desktop/Code/LincodeHackathon/Question 1/Images"
ListFiles = [f for f in listdir(ImagePath) if isfile(join(ImagePath, f))]
ListFiles.sort()

counter = 1
crop_counter = 1
for i in ListFiles:
    path = ImagePath+"/"+i
    print(path)
    img = cv2.imread(path)[:,:,::-1]
    # cv2.imshow("Image",img)
    plt.xticks([]), plt.yticks([])
    plt.imshow(img,cmap = "brg")
    plt.show(block=False)
    n = input("Test : ")
    if n == "q":
        continue
    elif n == "d":
        os.remove(path)
    elif n == "c":
        os.rename(path,"Crop"+str(crop_counter)+".jpg")
        crop_counter += 1
    elif n == "n":
        os.rename(path,str(counter)+".jpg")
        counter += 1
    
    # if cv2.waitKey(0) == ord('q'):
    #     break
    # elif cv2.waitKey(0) == ord('d'):
    #     os.remove(path)
    # elif cv2.waitKey(0) == ord('c'):
    #     os.rename(path,"Crop"+str(crop_counter)+".jpg")
    #     crop_counter += 1
    # elif cv2.waitKey(0) == ord("n"):
    #     os.rename(path,"Crop"+str(counter)+".jpg")
    #     crop_counter += 1