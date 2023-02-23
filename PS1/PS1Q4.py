import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import color, io

class Prob4():
    def __init__(self):
        """Load input color image indoor.png and outdoor.png here as class variables."""

        self.indoor = None
        self.outdoor = None
        ###### START CODE HERE ######
        self.indoor = cv2.imread("indoor.png")
        self.outdoor = cv2.imread("outdoor.png")
        ###### END CODE HERE ######

    
    def prob_4_1(self):
        """Plot R,G,B channels separately and also their corresponding LAB space channels separately"""
        
        ###### START CODE HERE ######
        # Bi, Gi, Ri = cv2.split(self.indoor)
        # Bo, Go, Ro = cv2.split(self.outdoor)
        # plt.imshow(Ri, cmap='gray')
        # plt.show()
        # plt.imshow(Gi, cmap='gray')
        # plt.show()
        # plt.imshow(Bi, cmap='gray')
        # plt.show()
        # plt.imshow(Ro, cmap='gray')
        # plt.show()
        # plt.imshow(Go, cmap='gray')
        # plt.show()
        # plt.imshow(Bo, cmap='gray')
        # plt.show()
        indoorLAB = cv2.cvtColor(self.indoor, cv2.COLOR_BGR2LAB)
        outdoorLAB = cv2.cvtColor(self.outdoor, cv2.COLOR_BGR2LAB)
        Li, Ai, Bi = cv2.split(indoorLAB)
        Lo, Ao, Bo = cv2.split(outdoorLAB)
        plt.imshow(Li, cmap='gray')
        plt.show()
        plt.imshow(Ai, cmap='gray')
        plt.show()
        plt.imshow(Bi, cmap='gray')
        plt.show()
        plt.imshow(Lo, cmap='gray')
        plt.show()
        plt.imshow(Ao, cmap='gray')
        plt.show()
        plt.imshow(Bo, cmap='gray')
        plt.show()
        ###### END CODE HERE ######
        return

    def prob_4_2(self):
        """
        Convert the loaded RGB image to HSV and return HSV matrix without using inbuilt functions. Return the HSV image as HSV. Make sure to use a 3 channeled RGB image with floating point values lying between 0 - 1 for the conversion to HSV.

        Returns:
            HSV image (3 channeled image with floating point values lying between 0 - 1 in each channel)
        """
        
        HSV = None
        ###### START CODE HERE ######
        imag = cv2.imread("inputPS1Q4.jpg")
        imag = imag.astype(np.double) / 255
        B, G, R = cv2.split(imag)
        V = np.max(imag, 2)
        m = np.min(imag, 2)
        C = V - m
        S = C / V
        S[V == 0] = 0
        H = np.zeros((V.shape))
        for i in range(V.shape[0]):
            for j in range(V.shape[1]):
                hprime = 0
                if C[i, j] == 0:
                    hprime = 0
                elif V[i, j] == R[i ,j]:
                    hprime = (G[i, j] - B[i, j]) / C[i, j]
                elif V[i, j] == G[i ,j]:
                    hprime = (B[i, j] - R[i, j]) / C[i, j] + 2
                elif V[i, j] == B[i ,j]:
                    hprime = (R[i, j] - G[i, j]) / C[i, j] + 4
        
                if hprime < 0:
                    H[i, j] = hprime / 6 + 1 
                else: 
                    H[i, j] = hprime / 6 
        HSV = cv2.merge([H, S, V])
        ###### END CODE HERE ######
        return HSV

        
if __name__ == '__main__':
    
    p4 = Prob4()
    p4.prob_4_1()
    HSV = p4.prob_4_2()
    cv2.imshow('HSV', HSV)
    cv2.waitKey()
    cv2.destroyAllWindows()





