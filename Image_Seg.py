# coding=gbk
'''
    IMAGE SEGMENTATION USING K-MEANS
    AUTHOR Li Hui

    command line arguments:
		`python imageSegmentation.py K inputImageFilename outputImageFilename`
     K is the number of classes which is greater than 3
'''

import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
import sys
# import cv2

#	Parse command-line arguments
#	sets K, inputName & outputName
if len(sys.argv) < 3:
	print("Error: Insufficient arguments, imageSegmentation takes three arguments")
	sys.exit()
else:
	K = int(sys.argv[1])
	if K < 3:
		print("Error: K has to be greater than 2")
		sys.exit()
	inputName = sys.argv[2]
	outputName = sys.argv[3]

iterations = 5

image = Image.open(inputName)
imageW = image.size[0]
imageH = image.size[1]

image_data = np.array(image)
# print('image_data shape:', image_data.shape)
row, col, chanel = image_data.shape
data_temp = image_data.reshape(-1, chanel)

dataVector_norm = data_temp / 256.0  # Normalization
# dataVector_norm = load_data('cherry.png')
# print('data shape:', data.shape)
# result: (250 000, 3)

# file_path = 'cherry.png'
# file_path = inputName

def distance(vecA, vecB):
    '''
    the squre of euler_distance between inputs
    params:
    input: vecA, vecB

    '''
    dist = (vecA - vecB) * (vecA - vecB).T
    return dist


def randCent(data, k):
    '''random initialize the centre of clusters
    input:  data(mat):训练数据
        k(int):类别个数
    output: centerids(mat):聚类中心
    '''
    n = np.shape(data)[1]  # the num of channels
    centerids = np.mat(np.zeros((k, n)))  # initialize K centre
    for j in range(n):
        minJ = np.min(data[:, j])
        rangeJ = np.max(data[:, j]) - minJ
        # initialize between the max and the min
        centerids[:, j] = minJ * np.mat(np.ones((k, 1))) + np.random.rand(k, 1) * rangeJ
    return centerids

# K = 4
# iterations = 1
centerids = randCent(dataVector_norm, K)

# image = Image.open(file_path)
# imageW = image.size[0]
# imageH = image.size[1]
pixelClusterAppartenance = np.ndarray(shape=(imageW * imageH), dtype=int)

# the iteration of K-means
for iteration in range(iterations):
    #	Set pixels to their cluster
    for idx, data in enumerate(dataVector_norm):
        distanceToCenters = np.ndarray(shape=(K))
        for index, center in enumerate(centerids):
            distanceToCenters[index] = distance(data.reshape(1, -1), center.reshape(1, -1))
        pixelClusterAppartenance[idx] = np.argmin(distanceToCenters)

    ##################################################################################################
    #	Check if a cluster is ever empty, if so append a random datapoint to it
    clusterToCheck = np.arange(K)  # contains an array with all clusters
    # e.g for K=10, array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    clustersEmpty = np.in1d(clusterToCheck, pixelClusterAppartenance)
    # ^ [True True False True * n of clusters] False means empty

    for index, item in enumerate(clustersEmpty):
        if item == False:
            pixelClusterAppartenance[np.random.randint(len(pixelClusterAppartenance))] = index
        # ^ sets a random pixel to that cluster as mentioned in the homework writeup
    ##################################################################################################

    for i in range(K):
        dataInCenter = []

        for index, item in enumerate(pixelClusterAppartenance):
            if item == i:
                dataInCenter.append(dataVector_norm[index])
        dataInCenter = np.array(dataInCenter)
        centerids[i] = np.mean(dataInCenter, axis=0)

    print("Centers Iteration num", iteration, ": \n", centerids)

for index, item in enumerate(pixelClusterAppartenance):
    dataVector_norm[index, 0] = centerids[item, 0] * 255
    dataVector_norm[index, 1] = centerids[item, 1] * 255
    dataVector_norm[index, 2] = centerids[item, 2] * 255

dataVector_norm = np.around(dataVector_norm).astype(int)

data_processed = dataVector_norm.reshape(imageW, imageH, 3)

plt.subplot(1, 2, 1)
plt.title('source img')
plt.imshow(image_data)
plt.subplot(1, 2, 2)
plt.title('Segmented img')
plt.imshow(data_processed)
plt.show()

data_temp = data_processed.astype(np.uint8)#conveter the uint32 to uint8
im = Image.fromarray(data_temp)# input must be a unit8 numpy.ndarray
im.save(outputName)





