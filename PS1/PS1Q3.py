import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import io


class Prob3():
    def __init__(self):
        """Load input color image inputPS1Q3.jpg here and assign it as a class variable"""
        self.img = None
        ###### START CODE HERE ######
        self.img = cv2.imread("inputPS1Q3.jpg")
        ###### END CODE HERE ######
    
    def rgb2gray(self, rgb):
        """
        Do RGB to Gray image conversion here. Input is the RGB image and you must return the grayscale image as gray

        Returns:
            gray: grayscale image
        """
        gray = None
        ###### START CODE HERE ######
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        ###### END CODE HERE ######
        return gray
        
    def prob_3_1(self):
        """
        Swap red and green color channels here, and return swapImg

        Returns:
            swapImg: RGB image with R and G channels swapped (3 channeled image with integer values lying between 0 - 255)
        """

        swapImg = None
        ###### START CODE HERE ######
        swapImg = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        R, G, B = cv2.split(swapImg)
        swapImg = cv2.merge([G, R, B])
        ###### END CODE HERE ######
        return swapImg

    
    def prob_3_2(self):
        """
        This function would simply call your rgb2gray function and return the grayscale image.

        Returns:
            grayImg: grayscale image (1 channeled image with integer values lying between 0 - 255)
        """
        grayImg = None
        ###### START CODE HERE ######
        grayImg = self.rgb2gray(self.img)
        ###### END CODE HERE ######
        return grayImg
    
    def prob_3_3(self):
        """
        Convert grayscale image to its negative.

        Returns:
            negativeImg: negative image (1 channeled image with integer values lying between 0 - 255)
        """
        negativeImg = None
        ###### START CODE HERE ######
        negativeImg = 255 - self.prob_3_2()
        ###### END CODE HERE ######
        return negativeImg
    
    def prob_3_4(self):
        """
        Create mirror image of gray scale image here.
        
        Returns:
            mirrorImg: mirror image (1 channeled image with integer values lying between 0 - 255)
        """
        mirrorImg = None
        ###### START CODE HERE ######
        mirrorImg = cv2.flip(self.prob_3_2(), 1)
        ###### END CODE HERE ######
        return mirrorImg
    
    def prob_3_5(self):
        """
        Average grayscale image with mirror image here.
        
        Returns:
            avgImg: average of grayscale and mirror image (1 channeled image with integer values lying between 0 - 255)
        """
        avgImg = None
        ###### START CODE HERE ######
        grayImg = self.prob_3_2()
        mirrorImg = self.prob_3_4()
        avgImg = (grayImg.astype(float) + mirrorImg.astype(float)) * 0.5
        avgImg = avgImg.astype(np.uint8)
        ###### END CODE HERE ######
        return avgImg
    
    def prob_3_6(self):
        """
        Create noise matrix N and save as noise.npy. Add N to grayscale image, clip to ensure that max value is 255.
        
        Returns:
            noisyImg, noise: grayscale image after adding noise (1 channeled image with integer values lying between 0 - 255)
            and the noise
        """
        noisyImg, noise = [None]*2
        ###### START CODE HERE ######
        grayImg = self.prob_3_2()
        noise = np.random.randint(0, 255, grayImg.shape)
        noisyImg = grayImg + noise
        np.clip(noisyImg, 0, 255, out=noisyImg)
        noisyImg = cv2.convertScaleAbs(noisyImg)
        np.save("noise.npy", noise)
        ###### END CODE HERE ######
        return noisyImg, noise
        
        
if __name__ == '__main__': 
    
    p3 = Prob3()

    swapImg = p3.prob_3_1()
    grayImg = p3.prob_3_2()
    negativeImg = p3.prob_3_3()
    mirrorImg = p3.prob_3_4()
    avgImg = p3.prob_3_5()
    noisyImg,_ = p3.prob_3_6()
    
    # cv2.imshow('Swap', swapImg)
    # cv2.imshow("Gray", grayImg)
    # cv2.imshow("Mirror", mirrorImg)
    # cv2.imshow("Negative", negativeImg)
    # cv2.imshow("Avg", avgImg)
    # cv2.imshow("Noisy", noisyImg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    




