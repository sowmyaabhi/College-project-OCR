import TextDetection as text
import cv2
import matplotlib.pyplot as plt
import time
import numpy as np 
import networkx as nx
import random

image=cv2.imread("1.jpg")

swt=text.SWT()
G=nx.Graph()

start=time.time()
SWTImage,rays=swt.getSWT(image,False,optimizeSpeed=False)
SWTImage=swt.SWTMeadianFilter(SWTImage,rays)



out=swt.normalizeFilter(SWTImage)


width=SWTImage.shape[1]
heigth=SWTImage.shape[0]

row_loc=np.zeros(SWTImage.shape[0]*SWTImage.shape[1])
componentImage=np.zeros((SWTImage.shape[0],SWTImage.shape[1],3))

def __outOfBoundary(X,Y,widthPix,heigthPix):
	if(X<0 or X>=widthPix or Y<0 or Y>=heigthPix):
		return True
	return False


for y in xrange(1,heigth):
	for x in xrange(1,width):
		currentPix=SWTImage.item(y,x)
		if currentPix>0:
			left=x-1
			up=y-1
			rigth=x+1
			
			leftPix=SWTImage.item(y,left)
			if(leftPix>=0 and (leftPix/currentPix<=3.0 or currentPix/leftPix<=3.0)):
				G.add_edge(y*width+x,y*width+left)
				row_loc.itemset((y*width+left),y)
			leftUpPix=SWTImage.item(up,left)
			if(leftUpPix>=0 and (leftUpPix/currentPix<=3.0 or currentPix/leftUpPix<=3.0)):
				G.add_edge(y*width+x,up*width+left)
				row_loc.itemset((up*width+left),up)
			upPix=SWTImage.item(up,x)
			if(upPix>=0 and (upPix/currentPix<=3.0 or currentPix/upPix<=3.0)):
				G.add_edge(y*width+x,up*width+x)
				row_loc.itemset((up*width+x),up)
			if(rigth<width):
				rightUpPix=SWTImage.item(up,rigth)
				if (rightUpPix>0 and (rightUpPix/currentPix<=3.0 or currentPix/rightUpPix<=3.0)):
					G.add_edge(y*width+x,int(up*width+rigth))
					row_loc.itemset(up*width+rigth,up)
			row_loc.itemset((y*width+x),y)

components=nx.algorithms.components.connected_components(G)

end=time.time()
res=(end-start)
print res



#print components
for component in components:
	color=np.array([int(random.random()*255),int(random.random()*255),int(random.random()*255)])
	for v in component:
		row=row_loc.item(v)
		col=v-row*width
		#print row,col,row_loc.item(v),v
		componentImage[row][col]=color#((int(row),int(col)),color)


componentImage=np.uint8(componentImage)
#plt.imshow(componentImage)
cv2.imshow("d",componentImage)
cv2.waitKey(0)
#plt.show()