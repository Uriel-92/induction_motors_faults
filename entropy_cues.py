# Script for calculate entropy cues
import numpy as np
import cv2

class entropyCues:
    def __init__(self, image):
        # The image needs to be a square matrix -> NxN
        if(image.shape[0] != image.shape[1]):
            self.image = cv2.resize(image, (250, 250), 
                               interpolation=cv2.INTER_AREA)
        else:
            self.image = image
            
        # Save entropy signal
        self.values = []
    
    def histogram(self, signal):
        # Array of size of the matrix length
        hist = np.zeros((np.max(signal) + 1,), dtype=np.float16)
        
        # Walk throught the signal
        for i in range(len(signal)):
            hist[signal[i]] += 1
            
        return hist/len(signal)
    
    def entropy(self):
        # Calculate the entropy
        width = self.image.shape[1]
        for i in range(width):
            signal = self.image[:,i]
            hist = self.histogram(signal)
            entropy = 0
            for j in range(len(hist)):
                if(hist[j] != 0):
                    entropy -= hist[j]*np.log2(hist[j])
            self.values.append(entropy)
        
        return np.array(self.values)
    
# Test passed :)