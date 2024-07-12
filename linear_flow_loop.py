# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 13:18:14 2024

@author: jcrompto
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 10:15:23 2024

@author: jcrompto
"""
# %% import raster and plotting tools
import geoutils as gu
import numpy as np
import matplotlib.pyplot as plt
import requests
import rasterio.plot 
import rasterio as rio
import xdem
from scipy import signal
from scipy import spatial as spt
from scipy import interpolate
import matplotlib.patches as mpatches
#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'qt')
import sys

# %% import scipy tools
from scipy.ndimage import gaussian_filter
from scipy.ndimage import median_filter

# %% import the the raster files created by another program
dd_rast = gu.Raster(r'C:\Users\jcrompto\Documents\code\python_scripts\jupyter_notebooks\remote_sensing\find_linears\saved_mtx\diffDEM.tif')
ddhs_rast = gu.Raster(r'C:\Users\jcrompto\Documents\code\python_scripts\jupyter_notebooks\remote_sensing\find_linears\saved_mtx\diffDEM_hs.tif')


# %% plot the raster data being imorted
ddhs = ddhs_rast.data
dd = dd_rast.data
plt.imshow(ddhs,cmap = 'Grays')

# %% set up some paramters
bcs = 5             ## this is the size of the center chip used to move through the linear trough
halfBcs = int(np.floor(bcs/2))
bckChipSize = 11    ## this is the size of the chip used to identify the background
halfBkgChip = int(np.floor(bckChipSize/2))
diffAnomThresh = 11  ## this is how far above the background the center pixel must be to qualify as a starting point
dmSz = 155          ## the size of the sampling window
gaussSigma=1.1      ## size of the standard deviation of the gaussian smooting window
distThresh = 1
minCrackLength = 10

# %% gaussian filter of the hillshade or detrended topography
gaussHS = gaussian_filter(ddhs, sigma=gaussSigma)
gH = np.copy(gaussHS[0:dmSz,0:dmSz])
#gH = np.copy(ddhs[0:dmSz,0:dmSz])
mean_gH = np.mean(gH)


# %% a chip is created that starts to scan across the window. when the center pixel of the chip is in sufficient contrast (large mean difference to the 
# average of the chip, we find a winner and start looking for a linear
lenLoop = 150   # this is how long you want to run before you find a mean difference that is acceptable
mean_bckgChip = np.zeros(lenLoop)
ctrZ =  np.zeros(lenLoop)
diffMeanM = np.zeros(lenLoop)
diffMean = 0
count = 0
crackElements = np.zeros((1,2))


for j in np.arange(1,dmSz-np.ceil(bckChipSize/2),2,dtype = int): # start the background chip search 1 in from the boundary so the small chip 
    for i in np.arange(1,dmSz-np.ceil(bckChipSize/2),dtype = int):
        
        cX = bckChipSize    # background chip size in X....choose an odd number
        cY = bckChipSize   # background chip size in Y
        intX_lf = int(i)   # left, right, up and down boundaries
        intX_rt = int(intX_lf+cX)
        intY_up = int(j)
        intY_dn = int(intY_up+cY)
        cX_c = np.floor(cX/2)
    
        bckgChip = np.copy(gH[intY_up:intY_dn,intX_lf:intX_rt])
        xCtr = int(intX_lf + (cX_c))
        yCtr = int(intY_up + (cX_c))
        ctr = np.copy(gH[yCtr,xCtr])    ## recheck that this is actually grabbing the center coordinate
    
        mean_bckgChip[i] = np.mean(bckgChip)
        ctrZ[i] = ctr
        diffMean = ctr - np.mean(bckgChip)
        diffMeanM[i] = ctr - np.mean(bckgChip)  
        
        if count==0:
            dCtr = 100
        if count>0:
            dCtr = np.sqrt(np.square(crackElements[:,0]-yCtr) + np.square(crackElements[:,1]-xCtr))
        # instY = np.where(yCtr==expandedCrackULong[:,0])
        # instX = np.where(yCtr==expandedCrackULong[:,1])
        # inst = np.intersect1d(instX,instY)
        # NoCrackInChip = inst.size==0
        
    
        if diffMean > diffAnomThresh and np.sum(dCtr<distThresh)==0:   # a cener pixel stands out above the average and no other cracks 
                                                          # have been previously identified in other pixels in the chup              
            count += 1
            print('you found a reasonable threshold')
            # plt.plot(mean_bckgChip)
            # # plt.plot(mean_bckgChip_ctr)
            # plt.plot(ctrZ)
            # plt.plot(diffMeanM)
    
            ## once you have found a qualifying pixel, make a chip in it's neighbourhood 
            # and find the max value and correspoding coordinates. upu dont want to use the bigger backgroung chip
            #because the bottom of the crack be way at the bottom of the chip, so you just use the top jalf of the chip
            chip_top  = np.copy(gH[intY_up:yCtr,intX_lf:intX_rt])
        
                
            zMax = np.max(chip_top)   # find where that chip has a max
            arB = gH==zMax
            ind_1 = np.where(arB)
            #plt.imshow(chip_1,cmap = 'Grays')
            xPos = np.squeeze(ind_1[1]) 
            yPos = np.squeeze(ind_1[0]) 
            dCtr = np.sqrt(np.square(crackElements[:,0]-yPos) + np.square(crackElements[:,1]-xPos))
            # make a new 3*3 chip arounf the new max,
            
            chip = np.copy(gH[yPos-1:yPos+2,xPos-1:xPos+2]) 
            #plt.imshow(chipN,cmap = 'Grays')
            #print(chipN)
    
            ## now that you;ve found the max in the neighbourhood of the qualifying pixel, start with that as the center and carve through the feature
            chip[1,1]=0
            maxMiddleY = np.copy(yPos)
            maxMiddleX = np.copy(xPos)
            #print('first chip = ', chip)
            
            #mumIts = 40
            #bcs = 5
            
            meanBIGMtx = np.mean(bckgChip)
            diffAnom = zMax - np.mean(bckgChip)
            diffAnomMtx = diffAnom
    
            k = 1
            gH_cp = np.copy(gH)
            
            while diffAnom > diffAnomThresh and np.sum(dCtr<distThresh)==0 and (xPos >= halfBcs and  xPos < dmSz - (halfBcs + 1) and yPos < dmSz - (halfBcs + 1)):
                
                print('you are starting to find the crack for iteration k =',k)
                print('the outer loop is at count =',count)
                print('i =',i)
                print('j =',j)
                print(chip)
                if k == 1:         ## since you're scanning top down, you want to make sure that you're working                                     
                    chip[0,:]=0    ## your way down through the line
                zMax = np.max(chip)
                arB = gH==zMax
                indNext = np.where(arB)
                
                gH_cp[yPos-1:yPos+2,xPos-1:xPos+2] = 0   # here you have found the max value in the chip and you don't want
                print('zMax = ',zMax)                    # to start circling around that chip, so you set all values to 0
                print('indNext =', indNext)
                ## here you are computing a new background of the large background size chip, if it straddles domain, use the old background
                if yPos > halfBkgChip and xPos > halfBkgChip and yPos < dmSz-1-halfBkgChip and xPos < dmSz-1-halfBkgChip:
                    bckgChip = np.copy(gH[int(yPos-halfBkgChip):int(yPos+halfBkgChip+1),
                                          int(xPos-halfBkgChip):int(xPos+halfBkgChip+1)])      
                meanBIG = np.mean(bckgChip)
                meanBIGMtx = np.hstack((meanBIGMtx,meanBIG))
                # if np.isnan(meanBIG):
                #     break
                print('meanBIG = ', meanBIG)
                
                yPos =  np.squeeze(indNext[0])
                xPos  = np.squeeze(indNext[1]) 
                #plt.imshow(gH)
                maxMiddleY = np.vstack((maxMiddleY,yPos))
                maxMiddleX = np.vstack((maxMiddleX,xPos))
                chip = np.copy(gH_cp[yPos-1:yPos+2,xPos-1:xPos+2])
                diffAnom = zMax - meanBIG
                #print(diffAnom)
                diffAnomMtx = np.hstack((diffAnomMtx,diffAnom))
                dCtr = np.sqrt(np.square(crackElements[:,0]-yPos) + np.square(crackElements[:,1]-xPos))
                k +=1
                #print(chip)
                
            plt.imshow(gH,cmap = 'Grays')
            if maxMiddleY.size>minCrackLength:
                #plt.disableAutoRange()
                plt.plot(maxMiddleX,maxMiddleY,'r-+')
               #plt.autoRange()
                crackElements = np.vstack((crackElements,np.hstack((maxMiddleY,maxMiddleX))))

expandedCrackU = np.unique(crackElements,axis=0)
expandedCrackU = expandedCrackU[1:]

# %% in this section of code you are taking all of the lines to 
##concatenate lines that are together and separating each line into 
##a list so that it becomes its own crack. doing this by finding the 
##distance between all pairs of points

yy_T,yy = np.meshgrid(expandedCrackU[:,0],expandedCrackU[:,0])
xx_T,xx = np.meshgrid(expandedCrackU[:,1],expandedCrackU[:,1])
diff_X2 = np.square(xx_T-xx)
diff_Y2 = np.square(yy_T-yy)

bigDist = np.sqrt(2*np.square(dmSz))

dist = np.sqrt(diff_X2 + diff_Y2)
dist[dist==0]=bigDist

crackCoordsOrd = np.array((yy_T[0,0],xx_T[0,0]))
numEl = np.shape(xx_T)[0]
lenLoop = numEl
minDistM = np.zeros(lenLoop)
beenCounted = np.ones(numEl)
#for counter in np.arange(numEl):
crackList = []
innerCrackList = np.zeros((1,2))
inCount = 0
    
n_j = 0   # this is the column of the distance matrix or the elements in the expandedCrackU array
for c_i in np.arange(numEl-1):
    n_i = np.where(dist[np.squeeze(n_j),:]==np.min(dist[n_j,:]))   # n_i is the row where the colum n_j is at a minimum, once you fund it, go down to the n_i column, such that the new n_j = n_i
    minDistM[c_i]=np.min(dist[n_j,:])
    beenCounted[n_j]=0
    newCoord = expandedCrackU[n_i]
    crackCoordsOrd = np.vstack((crackCoordsOrd,newCoord))
    innerCrackList = np.vstack((innerCrackList,newCoord))
    dist[n_i,n_j]=bigDist
    print('n_j, n_i =',n_j,np.squeeze(n_i))
    n_j = np.copy(np.squeeze(n_i))
    
    print('c_i=',c_i)
    print('minDist =',np.squeeze(minDistM[c_i]))
    print('sum of beenCounted =', np.sum(beenCounted))
    
    # if c_i == 272:
    #     sys.exit()
    
    if minDistM[c_i] >= 2:
        innerCrackList = innerCrackList[1:,:]
        crackList.append(innerCrackList) 
        innerCrackList = np.zeros((1,2))
        stillInY = np.multiply(beenCounted,expandedCrackU[:,0])
        n_j = np.where(stillInY==np.min(stillInY[stillInY!=0]))
        inCount+=1
        # if inCount == 4:
        #     print('inCount =4')
        #     #sys.exit()
        if np.size(n_j)>1:
            print('houston we have a problem')
            sys.exit()
            n_j = n_j[0]

innerCrackList = innerCrackList[1:,:]
crackList.append(innerCrackList)

# %% Here you are sorting the cracks, then 
crackListAve = []
crackListFit = []

numCracks = len(crackList)
#for crk in np.arange(numCracks):
crk = 5
crack = crackList[crk]
crack_s = np.sort(crack,axis=0)
c_x = crack_s[:,1]
c_y = crack_s[:,0]
plt.plot(c_x,c_y,'+')

c_xU = np.unique(c_x)
lx = np.size(c_xU)
meanY = np.zeros(lx)
for r in np.arange(lx):
    elX = np.where(c_x==c_xU[r])
    meanY[r] = np.mean(c_y[elX]) 

crackListAve.append(np.rot90(np.vstack((c_xU,meanY)),3))
plt.plot(c_xU,meanY,'o')
xnew = np.arange(c_x[0],c_x[-1]+1)
cspl= interpolate.CubicSpline(c_xU,meanY)
ynew = cspl(c_xU)
plt.plot(xnew,ynew,'r-')
pfit = np.polyfit(c_xU,meanY,3)
p = np.poly1d(pfit)
pFit = p(c_xU)
plt.plot(c_xU,pFit,'b-')
crackListFit.append(np.rot90(np.vstack((c_xU,pFit)),3))

#%%
plt.plot(expandedCrackU[:,1],expandedCrackU[:,0],'.')
