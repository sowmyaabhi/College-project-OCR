import TextDetection as text
import cv2
import matplotlib.pyplot as plt
import time
import numpy as np 
#import Algorithm as algo
import random

image=cv2.imread("1.jpg")
swt=text.SWT()
Fstart=time.time()
start=time.time()
SWTImage,rays=swt.getSWT(image,False,optimizeSpeed=True,medianFilterEnable=False)
end=time.time()
res=(end-start)
print res
start=time.time()
SWTImage=cv2.medianBlur(SWTImage,3)
end=time.time()
res=(end-start)
print res
start=time.time()
#components,row_loc=swt.connected_componentsRAY(SWTImage,rays)
components,row_loc=swt.connected_components(SWTImage)
#components,row_loc=algo.connected_components(SWTImage)
end=time.time()
res=(end-start)
print res
Fend=time.time()
res=(Fend-Fstart)
print res

width=image.shape[1]
componentImage=np.zeros((SWTImage.shape[0],SWTImage.shape[1],3))
for component in components:
	g=int(random.random()*255)
	b=int(random.random()*255)
	r=int(random.random()*255)
	for v in component:
		row=row_loc.item(v)
		col=v-row*width
		#print row,col,row_loc.item(v),v
		componentImage.itemset((row,col,0),g)
		componentImage.itemset((row,col,1),b)
		componentImage.itemset((row,col,2),r)

componentImage=np.uint8(componentImage)
plt.imshow(componentImage)
#cv2.imshow("d",componentImage)
#cv2.waitKey(0)
plt.show()