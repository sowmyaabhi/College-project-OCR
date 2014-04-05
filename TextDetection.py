import cv2
import numpy as np 
import math
import networkx as nx

class Point2D:
	x=None
	y=None
	SWT=None
	def __init__(self,x,y):
		self.x=x
		self.y=y

class Ray:
	p=None
	q=None
	points=None


class SWT():

	def __outOfBoundary(self,X,Y,widthPix,heigthPix):
		if(X<0 or X>=widthPix or Y<0 or Y>=heigthPix):
			return True
		return False
		
	def __getLength(self,p,q):
		return (math.sqrt(math.pow(p.x-q.x,2)+math.pow(p.y-q.y,2)))

	def __clean_cos(self,cos_angle):
		return min(1,max(cos_angle,-1))

	def __validAngle(self,sineA,cosineA,sineB,cosineB):
		"""
			API Usage:		
							validAngle(self,sineA,cosineA,sineB,cosineB)
				
			Purpose:		To check whether the angle is valid or not.
		
			Parameters:	
				sineA:		Sine value of the angle A
	
				cosineA:	Cosine value of the angle A

				sineB:		Sine value of the angle B

				cosineB:	Cosine value of the angle B	
		"""
		#acos returns the principal value of the arc cosine of x, expressed in radians.
		#In trigonometrics, arc cosine is the inverse operation of cosine.
		angle=math.acos(self.__clean_cos(cosineA*cosineB+sineA*sineB))
		angle=math.fabs(angle)
		if(angle>math.pi/6.0):
			return True
		return False

	def __validLength(self,length,width,heigth,optimizeSpeed):
		"""
			API Usage:	validLength(self,length,width,heigth,optimizeSpeed)

			Purpose:	To check whether the length is valid or not

			Parameters:	
				length:	length of an stroke

				width:	width of an input image

				heigth:	heigth of an input image
		"""
		l=min(width,heigth)*15/100.0
		if length>l and optimizeSpeed:
			return False
		return True

	def normalizeFilter(self,SWTImage):
		output=np.zeros((SWTImage.shape[0],SWTImage.shape[1]))
		maxVal=SWTImage.max()
		minVal=SWTImage.min()
		dif=maxVal-minVal
		output[SWTImage<0]=1
		output[SWTImage>=0]=((SWTImage[SWTImage>=0])-minVal)/float(dif)
		return output

	def SWTMeadianFilter(self,SWTImage,rays):
		for r in rays:
			for p in r.points:
				p.SWT=SWTImage.item(p.y,p.x)
			r.points.sort(cmp = lambda x, y: cmp(x.SWT, y.SWT))
			median=(r.points[len(r.points)/2]).SWT
			for p in r.points:
				SWTImage.itemset((p.y,p.x),min(median,p.SWT))
		return SWTImage

	def __SWT(self,edgeImg,gradientX,gradientY,LIGHT_TO_DARK,medianFilterEnable,optimizeSpeed,prec=1):
		"""
			API Usage:
				__SWT(self,edgeImg,gradientX,gradientY,LIGHT_TO_DARK,optimizeSpeed,[prec=1])

			Purpose:
				Stroke width transform find the stroke length with some filtering improvement.

			How it works?
				First it takes an image as the input.Then it finds the gradient value for 
			each of the pixel of the image.It finds the gradient value using the Sobel Operator.
			Further the cos and sin value are given by G_x/mag and G_y/mag respectively.For each
			edge pixel P, it identifies the gradient direction such that the gradient direction 
			should be perpendicular to the pixel.Then it traverse in the gradient direction until
			another edge pixel is found.If no edge pixel is found, then the gradient direction is 
			neglected.

			Parameters:
				edgeImg:	It is the output edge image obtained from the Canny Edge Detection
					algorithm.
				
				gradientX:	It is the gradient value obtained from the Sobel operator in the X direction.
		
				gradientY:	It is the gradient value obtained from 	the Sobel operator in the Y direction.
		
				LIGHT_TO_DARK(Boolean):	If set to True, it means Light text on Dark background. if set
					to False, it means Dark text on Light background.

				prec(optional by deafult 1):	prec can be set by user. If not set by user, its value 
					by default is 1.  			
		"""
		widthPix=edgeImg.shape[1]
		heigthPix=edgeImg.shape[0]
		width=widthPix
		heigth=heigthPix
		SWTImage=np.zeros(edgeImg.shape)
		SWTImage.fill(-1)
		rays=[]
		for y in xrange(heigthPix):
			for x in xrange(widthPix):
				if edgeImg.item(y,x)>0:
					ray=Ray()
					G_x=gradientX.item(y,x)
					G_y=gradientY.item(y,x)
					magnitude=math.sqrt(math.pow(G_x,2)+math.pow(G_y,2))
					if LIGHT_TO_DARK:
						sine=G_y/magnitude
						cosine=G_x/magnitude
					else:
						sine=-G_y/magnitude
						cosine=-G_x/magnitude
					newX=x
					newY=y
					p=Point2D(x,y)
					ray.p=p
					points=[]
					while(True):
						newX=newX+prec*cosine
						newY=newY+prec*sine
						intermediat=Point2D(newX,newY)
						if self.__outOfBoundary(newX,newY,widthPix,heigthPix):
							break
						if(edgeImg.item(newY,newX)>0):
							q=Point2D(newX,newY)
							points.append(q)
							ray.q=q
							G_x=gradientX.item(q.y,q.x)
							G_y=gradientY.item(q.y,q.x)

							magnitudeQ=math.sqrt(math.pow(G_x,2)+math.pow(G_y,2))
							if LIGHT_TO_DARK:
								sineQ=G_y/magnitudeQ
								cosineQ=G_x/magnitudeQ
							else:
								sineQ=-G_y/magnitudeQ
								cosineQ=-G_x/magnitudeQ

							length=self.__getLength(p,q)
							if self.__validAngle(sine,cosine,sineQ,cosineQ):
								for point in points:
									if SWTImage.item(point.y,point.x)>=0:
										SWTImage.itemset((point.y,point.x),min(length,SWTImage.item(point.y,point.x)))
									else:
										SWTImage.itemset((point.y,point.x),length)
							ray.points=points
							if medianFilterEnable:
								rays.append(ray)						
							break
						length=self.__getLength(p,intermediat)
						if not self.__validLength(length,width,heigth,optimizeSpeed):
							break
						points.append(intermediat)
		return (SWTImage,rays)

	def connected_components(self,SWTImage):
		width=SWTImage.shape[1]
		heigth=SWTImage.shape[0]
		G=nx.Graph()
		row_loc=np.zeros(SWTImage.shape[0]*SWTImage.shape[1])

		for y in xrange(1,heigth):
			for x in xrange(1,width):
				currentPix=SWTImage.item(y,x)
				if currentPix>0:
					left=x-1
					up=y-1
					rigth=x+1

					leftPix=SWTImage.item(y,left)
					if(leftPix>0 and (leftPix/currentPix<=3.0 or currentPix/leftPix<=3.0)):
						G.add_edge(y*width+x,y*width+left)
						row_loc.itemset((y*width+left),y)

					leftUpPix=SWTImage.item(up,left)
					if(leftUpPix>0 and (leftUpPix/currentPix<=3.0 or currentPix/leftUpPix<=3.0)):
						G.add_edge(y*width+x,up*width+left)
						row_loc.itemset((up*width+left),up)

					upPix=SWTImage.item(up,x)
					if(upPix>0 and (upPix/currentPix<=3.0 or currentPix/upPix<=3.0)):
						G.add_edge(y*width+x,up*width+x)
						row_loc.itemset((up*width+x),up)

					if(rigth<width):
						rightUpPix=SWTImage.item(up,rigth)
						if (rightUpPix>0 and (rightUpPix/currentPix<=3.0 or currentPix/rightUpPix<=3.0)):
							G.add_edge(y*width+x,up*width+rigth)
							row_loc.itemset(up*width+rigth,up)

					row_loc.itemset((y*width+x),y)

		components=nx.algorithms.components.connected_components(G)
		return components,row_loc

	def getSWT(self,image,DARK_ON_LIGHT,preprocessed=False,cannyThreshold=0,optimizeSpeed=True,medianFilterEnable=False):
		"""
			API Usage:
			        getSWT(self,image,DARK_ON_LIGHT,[preprocessed=False,cannyThreshold=0,[optimizeSpeed=True]])
				
			Purpose:
				This function allows optional parameters preprocessed and optimizeSpeed.

			Parameters:
				image:	This is the input image. If preprocessed is disabled(preprocessed=False by default
					then pass an GBR image as an input.If preprocessed is enabled, then pass an Gray scale 
					image as an input.
			
				DARK_ON_LIGHT(Boolean):	If set to True, it means Dark text on Light background. If set to False,
					it means Light text on Dark background.
	
				preprocessed(optional by default preprocessed=False):	If the preprocessed parameter is set to 
					True user can perform the preprocessing according to his requirements. If the preprocessed 
					is set to False, then the default preprocessing is done.In default preprocessing, it performs 
					OTSU Threshold which finds the optimal threshold value.It also performs Gaussian Blur, i.e. 
					Gaussian Filtering.OTSU's Thresholding is done after the Gaussian Filtering. Median and Gaussian
					removes the unwanted noise in the image.
			
				cannyThreshold(optional):	If preprocessed is set to True, then you should set the cannyThreshold 
					value. If preprocessed is set to False then cannyThreshold is set by OTSU threshold.
		"""
		if not(preprocessed):
			image = cv2.medianBlur(image,5)
			image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
			gaussianImg=cv2.GaussianBlur(image,(5,5),0)
			thresholdValue,_ = cv2.threshold(gaussianImg,0,255,cv2.THRESH_OTSU)
			edgeImg=cv2.Canny(gaussianImg,0,thresholdValue)
		else:
			gaussianImg=image
			edgeImg=cv2.Canny(gaussianImg,0,cannyThreshold)
		
		gradientX = cv2.Sobel(gaussianImg,cv2.CV_64F, 1, 0)
		gradientY = cv2.Sobel(gaussianImg,cv2.CV_64F, 0, 1)
		SWTImage,rays=self.__SWT(edgeImg,gradientX,gradientY,DARK_ON_LIGHT,medianFilterEnable,optimizeSpeed)
		if medianFilterEnable:
			SWTImage=self.SWTMeadianFilter(SWTImage,rays)
		return SWTImage