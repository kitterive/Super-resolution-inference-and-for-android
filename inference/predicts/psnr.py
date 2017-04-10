import numpy as np
import cv2 as cv
import math
from skimage.measure import structural_similarity as ssim

def psnr(img1, img2):
    mse = np.mean((img1-img2)**2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
# base_name = "reference"
# ref_img = cv.imread(base_name+".png")
# in_img = cv.imread(base_name+"_cubic.png")
# out_img = cv.imread(base_name+"_out.png")

base_name = "jk9"
ref_img = cv.imread(base_name+".jpg")
in_img = cv.imread(base_name+"_cubic.png")
out_img = cv.imread(base_name+"_out.png")


 # calculate psnr
test_gray = cv.cvtColor(ref_img,cv.COLOR_RGB2GRAY)
cubic_gray = cv.cvtColor(in_img,cv.COLOR_RGB2GRAY)
out_gray = cv.cvtColor(out_img,cv.COLOR_RGB2GRAY)
    
print test_gray.shape,  cubic_gray.shape,  out_gray.shape

psnr_cubic = psnr(test_gray,cubic_gray)
psnr_output = psnr(test_gray,out_gray)

ssim_cubic = ssim(test_gray,cubic_gray)
ssim_output = ssim(test_gray,out_gray)
    
print ("cubic interpolation psnr is:" + str(psnr_cubic) +";     " + "predict psnr is:" + str(psnr_output))
print ("cubic interpolation ssim is:" + str(ssim_cubic) +";     " + "predict ssim is:" + str(ssim_output))
