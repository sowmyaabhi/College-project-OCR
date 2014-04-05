import TextDetection as text
import cv2
import matplotlib.pyplot as plt
import time
import numpy as np 

import random

image=cv2.imread("2.jpg")

swt=text.SWT()

start=time.time()
SWTImage=swt.getSWT(image,False,optimizeSpeed=True,medianFilterEnable=False)
#out=swt.normalizeFilter(SWTImage)

componentImage=np.zeros((SWTImage.shape[0],SWTImage.shape[1],3))
components,row_loc=swt.connected_components(SWTImage)

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
plt.imshow(componentImage)
#cv2.imshow("d",componentImage)
#cv2.waitKey(0)
plt.show()