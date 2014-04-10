import os
import sys
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import math
from cPickle import dump, HIGHEST_PROTOCOL,load
from scipy.cluster.vq import vq,kmeans
import matplotlib.pyplot as plt

def get_immediate_subdirectories(dir):
	"""
		this function return the immediate subdirectory list
		eg:
			dir
				/subdirectory1
				/subdirectory2
				.
				.
		return ['subdirectory1',subdirectory2',...]
	"""

	return [name for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))]


def getWordsInImages(imgFilesPath,masksPath=None):
	"""
		this function return dictonary of SIFT descriptor (simular to words in text, descriptor in image)
		they take image files path and masks path(optional)
	"""
	t=sys.argv[2]+"/"+"SIFT.file"
	if not(os.path.isfile(t)):
		dictWordsImages={}
		sift=cv2.SIFT()
		for imgFilepath in imgFilesPath.keys():
			if imgFilepath.endswith((".png",".jpeg",".JPG",".JPEG",".jpg",".PNG",".pgm",".PGM",".TIF",".TIFF", ".tif", ".tiff",)):
				img=cv2.imread(imgFilepath)
				imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
				if masksPath!=None:
					imgMask=cv2.imread(masksPath[imgFilepath])
					imgMaskGray=cv2.cvtColor(imgMask,cv2.COLOR_BGR2GRAY)
				else:
					imgMaskGray=None
				kp, des = sift.detectAndCompute(imgGray,imgMaskGray)
				dictWordsImages[imgFilepath]=des
		file=open(t,"w")
		dump(dictWordsImages,file,protocol=HIGHEST_PROTOCOL)
	else:
		file=open(t,"r")
		dictWordsImages=load(file)
	return dictWordsImages

def dict2numpy(dict):
	"""
		this function return array from dictonary
	"""
	nkeys = len(dict)
	array = np.zeros((nkeys * PRE_ALLOCATION_BUFFER, 128))
	pivot = 0
	for key in dict.keys():
		value = dict[key]
		nelements = value.shape[0]
		while pivot + nelements > array.shape[0]:
			padding = zeros_like(array)
			array = vstack((array, padding))
		array[pivot:pivot + nelements] = value
		pivot += nelements
	array = np.resize(array, (pivot, 128))
	return array

def computeCentroids(k,max_iter=1000):
	"""
		this function compute centroid (called visual words) using k-means algorithm
	"""
	t=sys.argv[2]+"/"+"CENTROID.file"
	if not(os.path.isfile(t)):
		#centroids, distortion = kmeans(arrayOfWordsImage,k) used in case of scpy
		cluster=MiniBatchKMeans(init='k-means++', n_clusters=k,max_iter=max_iter,init_size=3*k)
		cluster.fit(arrayOfWordsImage)
		centroids = cluster.cluster_centers_
		file=open(t,"w")
		dump(centroids,file,protocol=HIGHEST_PROTOCOL)
	else:
		file=open(t,"r")
		centroids=load(file)
	return centroids

def computeHistrogram(centroids,dictWordsImages):
	"""
		this function compute the histrogram for visual words
	"""
	allwordVocabularyHistrogram={}
	for name in dictWordsImages.keys():
		code,dist=vq(dictWordsImages[name],centroids)
		#print code.shape
		#print centroids.shape
		#print dictWordsImages[name].shape
		#plt.figure(1)
		#plt.subplot(211)
		#plt.title("descriptors")
		#plt.imshow(dictWordsImages[name])
		#plt.subplot(212)
		#plt.title("codebook")
		#print code
		#plt.imshow(centroids[code[:]])
		#plt.show()
		allwordVocabularyHistrogram[name], bin_edges = np.histogram(code,bins=xrange(centroids.shape[0] + 1),normed=True)
	return allwordVocabularyHistrogram










if len(sys.argv)<3:
	print "the dataset folder with subfolder name as label, not provied"
	sys.exit(0)

if sys.argv[2][-1]=="/":
	sys.argv[2]=sys.argv[2][:len(sys.argv[2])-1]

labels=get_immediate_subdirectories(sys.argv[2])
pathToDatasetDir=[]
imgFilesPathsWithLabels={}
PRE_ALLOCATION_BUFFER=1000

print "------------------------------------------------------"
print "label are:"+str(labels)
print "------------------------------------------------------"

for dirList in labels:
		datasetDir=sys.argv[2]+"/"+dirList
		for file in os.listdir(datasetDir):
			imgFilesPathsWithLabels[datasetDir+"/"+file]=dirList
masksPath={}

print "Generation SIFT words"
print "------------------------------------------------------"
try:
	if sys.argv[3] and sys.argv[4]:
		for dirList in labels:
			masksDir=sys.argv[4]+"/"+dirList
			for file in os.listdir(masksDir):
				masksPath[sys.argv[2]+"/"+dirList+"/"+file]=masksDir+"/"+file
	dictWordsImages=getWordsInImages(imgFilesPathsWithLabels,masksPath)
except IndexError:
	print "No masks avilable"
	print "------------------------------------------------------"
	dictWordsImages=getWordsInImages(imgFilesPathsWithLabels)

arrayOfWordsImage=dict2numpy(dictWordsImages)
print "the number of words:"+str(dict2numpy(dictWordsImages).shape[0])
print "------------------------------------------------------"
print "Generation vocabulary or visual words using clustering"
print "------------------------------------------------------"
nclusters=int(50)
print "number of cluster="+str(nclusters)
centroids=computeCentroids(nclusters)
print "------------------------------------------------------"
print "Coumputing Histrogram from centroids"
allwordVocabularyHistrogram=computeHistrogram(centroids,dictWordsImages)
print "------------------------------------------------------"
print "write the histogram.train"
file=open(sys.argv[2]+"/"+"histrogram.out","w")
for name in allwordVocabularyHistrogram.keys():
	#################################################################imgId=imgFilesPathsWithLabels[name]
	imgId=name.split("/")[3]
	d=map(str,(allwordVocabularyHistrogram[name]).tolist())
	print >>file,imgId," ".join(d)
print "------------------------------------------------------"