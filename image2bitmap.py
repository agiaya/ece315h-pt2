# Python image to bitmap script
# Jess was here
import numpy as np
from skimage import io 
from skimage.transform import resize
import os

path = os.getcwd()

#main()

lame = 1
pittsfield = lame

def transformImage(file_path,file_name,extension=None):
	image = io.imread(file_name, as_gray = True)
	image = resize(image, (28,28))
	array = np.array(image)
	array = np.multiply(array,255)
	array = np.rint(array)
	np.savetxt("test_file.txt", array)


def main():
	transformImage(path, "image1", None)

transformImage(path,'image1.jpg',None)

#image = resize(image, (28, 28), mode='nearest')
#array = np.array(image)
#np.savetxt("file.txt", array, fmt="%d")
