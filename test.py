import os 
import cv2
import torch
import shutil
import numpy as np
from time import time
import torch.nn.functional as F
from kernal import togray,laplacian,gridImg,gaussion,conv2d

def lap_det(img,r=11,c=7):
    h,w = img.shape[:2]
    st = time()
    tensor = None
    if torch.cuda.is_available():
        tensor = torch.from_numpy(img).cuda().float()  
    else:
        tensor = torch.from_numpy(img).float()
    mh,mw = (h-8)%r, (w-8)%c
    hl,hr = mh//2,mh-mh//2
    wt,wb = mw//2,mw-mw//2
    crop = tensor[hl+4:h-hr-4,wt+4:w-wb-4]
    input = togray(crop)
    data = gridImg(input,r,c)
    gass_w = gaussion(3,5)
    weight = laplacian(0)
    if torch.cuda.is_available():
        gass_w = gass_w.cuda()
        weight = weight.cuda()
    out = torch.abs(conv2d(conv2d(data,gass_w),weight))
    var = out.reshape(out.size(0),-1).var(1)
    print("time: ",time()-st)
    return var
    
def get_blur(img_path,save_path="blur"):
    fs = os.listdir(img_path)
    for f in fs:
        path = os.path.join(img_path,f)
        img = cv2.imread(path)
        var = torch.sort(lap_det(img))[0]
        mean_max = ((var-var.mean())**2)[-10:].mean().cpu().numpy()
        value = var[:50].mean().cpu().numpy()
        print(mean_max,value)
        if value<1.0 and mean_max<0.116:
            shutil.move(path,os.path.join(save_path,f))
            
if __name__ == "__main__":
    img_path = "images"
    save_path = "blur"
    get_blur(img_path,save_path)