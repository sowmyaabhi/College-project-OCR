import cv2
import numpy as np
import math
import matplotlib.pyplot as plt 


DARK_ON_LIGHT=0


class Point2d:
	x=None
	y=None
	SWT=None
	def getIt(self):
		"""
			get the x and y value as tuple
		"""
		return (x,y)

class Ray:
	p=Point2d()
	q=Point2d()
	points=[]

def textDetection(img,dark_on_light=False):
	"""
		this perform image processing for text detection
	"""
	img = cv2.medianBlur(img,5)
	#convet to grayscale
	grayImg=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	#detect edge in image
	edgeImg=cv2.Canny(grayImg,175,320)
	#gaussianImg=cv2.GaussianBlur(grayImg,(5,5),0)
	gradientX = cv2.Sobel(grayImg,cv2.CV_64F, 1, 0)
	gradientY = cv2.Sobel(grayImg,cv2.CV_64F, 0, 1)

	(SWTImage,ray)=SWT(edgeImg,gradientX,gradientY,True)
	SWTImage=SWTMeadianFilter(SWTImage,ray)
	SWTImage=normalizeFilter(SWTImage)
	plt.imshow(SWTImage,cmap="gray")
	plt.show()

def SWT(edgeImg,gradientX,gradientY,DARK_ON_LIGHT):
	"""
		Stroke Width Transform
	"""
	rays=[]
	prec=0.05
	height=edgeImg.shape[0]
	width=edgeImg.shape[1]
	SWTImage=np.empty((height,width))
	SWTImage.fill(-1)

	for row in xrange(height):
		for col in xrange(width):
			if(edgeImg[row][col]>0):# and gradientX[row][col]>0 and gradientY[row][col]>0):
				r=Ray()
				p=Point2d()
				p.x=col
				p.y=row
				r.p=p
				points=[]
				points.append(p)
				curX=float(col)+0.5
				curY=float(row)+0.5
				curPixX=col
				curPixY=row
				G_x=gradientX[row][col]
				G_y=gradientY[row][col]
				mag=math.sqrt(math.pow(G_x,2)+math.pow(G_y,2))
				if(DARK_ON_LIGHT):
					G_x=-G_x/mag
					G_y=-G_y/mag
				else:
					G_x=G_x/mag
					G_y=G_y/mag
				while(True):
					curX+=G_x*prec
					curY+=G_y*prec
					if((math.floor(curX)!=curPixX) or (math.floor(curY)!=curPixY)):
						curPixX=math.floor(curX)
						curPixY=math.floor(curY)
						if((curPixX<0)or(curPixX>=width)or(curPixY<0)or(curPixY>=height)):
							break
						pnew=Point2d()
						pnew.x=curPixX
						pnew.y=curPixY
						points.append(pnew)
						if(edgeImg[curY][curX]>0):
							r.q=pnew
							G_xt=gradientX[curPixY][curPixX]
							G_yt=gradientY[curPixY][curPixX]
							mag=math.sqrt(math.pow(G_xt,2)+math.pow(G_yt,2))

							if(DARK_ON_LIGHT):
								G_xt=-G_xt/mag
								G_yt=-G_yt/mag
							else:
								G_xt=G_xt/mag
								G_yt=G_yt/mag

							if ( math.fabs(G_x * -G_xt + G_y * -G_yt)<1 and math.acos(G_x * -G_xt + G_y * -G_yt) <= math.pi/2.0 ):
								length=math.sqrt((float(r.q.x) - float(r.p.x))*(float(r.q.x) - float(r.p.x)) + (float(r.q.y) - float(r.p.y))*(float(r.q.y) - float(r.p.y)))
								for pit in points:
									if(SWTImage[pit.y][pit.x]<0):
										SWTImage[pit.y][pit.x]=length
									else:
										SWTImage[pit.y][pit.x]=int(min(length,SWTImage[pit.y][pit.x]))

								r.points = points
								rays.append(r)
							break
	return (SWTImage,rays)


def SWTMeadianFilter(SWTImage,ray):
	for r in ray:
		for p in r.points:
			p.SWT=SWTImage[p.y][p.x]
		r.points.sort(cmp = lambda x, y: cmp(x.SWT, y.SWT))
		median=(r.points[len(r.points)/2]).SWT
		for p in r.points:
			SWTImage[p.y][p.x]=int(min(median,p.SWT))
	return SWTImage

def normalizeFilter(SWTImage):
	output=np.zeros((SWTImage.shape[0],SWTImage.shape[1]))
	maxVal=SWTImage.max()
	minVal=SWTImage.min()
	dif=maxVal-minVal
	output[SWTImage<0]=1
	output[SWTImage>=0]=(SWTImage[SWTImage>=0]-minVal)/float(dif)
	return output


img=cv2.imread("2.jpg")
textDetection(img,0)