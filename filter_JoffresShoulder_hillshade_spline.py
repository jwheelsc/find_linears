#!/usr/bin/env python
# coding: utf-8

# in this script, lidar is imported, gridded, an interpolated with a spline
# In[210]:

import geoutils as gu
import numpy as np
import matplotlib.pyplot as plt
import requests
import rasterio.plot 
import rasterio as rio
import xdem
from scipy.interpolate import RegularGridInterpolator

# %%
filename = r'C:\Users\jcrompto\Documents\remote_sensing\lidar\joffre\LiDAR Raster\20_4031_01_1m_DTM_CSRS_z10_ellips.tif'
rast = gu.Raster(filename)

# In[]


#llx = 536802; lly = 5578395; urx = 537774; ury = 5579407;
llx = 537000; lly = 5579000; urx = 537200; ury = 5579200;
rast.crop((llx,lly,urx,ury),inplace = True)
rast.show(ax = "new",cmap = "Greys_r")


# here you are simply plotting the data above but in 3D view and downsampled by dx

# In[197]:

rastData = rast.data
t = np.linspace(np.min(rastData),np.max(rastData),983664)

xs, ys = np.shape(rastData)
x = np.linspace(1,xs,xs)
y = np.linspace(ys,1,ys)
xg,yg = np.meshgrid(x,y)

dx = 20

xgds = xg[::dx,::dx]
ygds = yg[::dx,::dx]
rdds = rastData[::dx,::dx]

fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
#ax.scatter(xgds.ravel(), ygds.ravel(), rdds.ravel(), c = rdds, cmap = "Grays", s=60,  label='data', marker = '.')
#ax.scatter(xgds, ygds, rdds, c = rdds, cmap = "Grays", s=60,  label='data', marker = '.',alpha=0.4)

ax.plot_wireframe(np.rot90(xgds),np.rot90(ygds),(rdds), rstride=3, cstride=3,
                  alpha=0.4, color='m', label='linear interp')
#ax.view_init(-40,60)


# in this cell, I am using an interpolator on the downscaled data, then using the smnoothed interpolation to resample at the original resolution

# In[198]:


xs, ys = np.shape(rastData)
x = np.linspace(1,xs,xs)
dx = 20
xds = x[::dx]
y = np.linspace(ys,1,ys)
yds = y[::dx]
xgds,ygds = np.meshgrid(xds,yds)
rdds = rastData[::dx,::dx]
interpDat = RegularGridInterpolator((xds,yds),rdds,method='cubic',bounds_error=False,)
int1m = interpDat((xg,yg))
dxCubic = np.fliplr(interpDat.values)

fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
ax.plot_wireframe(np.rot90(xgds),np.rot90(ygds),interpDat.values, rstride=3, cstride=3,
                  alpha=0.4, color='m', label='linear interp')
#ax.plot_wireframe(np.rot90(xg),np.rot90(yg),(interpDat.values), rstride=3, cstride=3,
#                  alpha=0.8, color='b', label='linear interp')
#ax.view_init(-40,60)


# In[214]:


intCubic = np.fliplr(np.rot90(int1m,-1))
diffDEM = rastData-intCubic


# In[215]:


fig, (ax) = plt.subplots(1,4)
ax[0].imshow(dxCubic,cmap = 'jet',extent = [0,xs,0,ys])
ax[0].set_title('20m spline')
ax[1].imshow(rastData,cmap = 'jet')
ax[1].set_title('data')
ax[2].imshow(intCubic,cmap = 'jet',extent = [0,xs,0,ys])
ax[2].set_title('resampled from interpolated')
ax[3].imshow(diffDEM,cmap = 'jet',extent = [0,xs,0,ys])
ax[3].set_title('diff DEM')


# In[204]:


dVals = diffDEM.ravel()
dBin = 0.25
binL = np.arange(-8,9,dBin)
binC = binL[0:-1]+(dBin/2)
pdf,indx = np.histogram(dVals[~np.isnan(dVals)],bins=binL)
plt.plot(binC,pdf)


# In[221]:


hs_rescale = exposure.rescale_intensity(hs, in_range=(p2, p98))


# In[216]:


hs,slope = hillshade(diffDEM,270,180)
p2, p98 = np.percentile(hs, (2, 90))
hs_rescale = exposure.rescale_intensity(hs, in_range=(p2, p98))
thresh = threshold_otsu(hs_rescale)
hs_thresh = hs_rescale > thresh
hs_rescale[hsThresh] = np.mean(hs_rescale.ravel())
hs_rescale[hsThresh] = 1

fig, (ax) = plt.subplots(2,2,figsize=(14,14))
ax[0,0].imshow(hs,cmap = 'Grays')
#ax[0,0].plot(hs[200,:] + hs[200,:],alpha=0.5,linewidth=0.8,color = 'red')
#dat1 = diffDEM[200,:]*20
#dat1 = dat1[~np.isnan(dat1)]
#ax[0,0].plot(dat1 + (200-np.mean(dat1)),alpha=0.5,linewidth=0.8,color = 'blue')
ax[0,0].set_title('hs')
ax[0,1].imshow(hs_rescale,cmap = 'Grays')
ax[0,1].plot(hs_rescale[200,:]+200,alpha=1)
ax[0,1].set_title('hs stretch')
ax[1,0].imshow(hs_thresh,cmap = 'binary')
ax[1,0].set_title('hs_thresh')
ax[1,1].imshow(hs_rescale,cmap = 'binary')
ax[1,1].set_title('hs rescale')


# In[222]:


dd_rast = rast.copy(new_array=diffDEM)
dd_rast.save(r'C:\Users\jcrompto\Documents\code\python_scripts\jupyter_notebooks\remote_sensing\find_linears\saved_mtx\diffDEM.tif')


# In[223]:


ddhs_rast = rast.copy(new_array=hs)
ddhs_rast.save(r'C:\Users\jcrompto\Documents\code\python_scripts\jupyter_notebooks\remote_sensing\find_linears\saved_mtx\diffDEM_hs.tif')


# In[175]:


fig,ax = plt.subplots(1,1,figsize=(10,10))
dat1 = diffDEM[200,:]
dat1 = dat1[~np.isnan(dat1)]
dat1N =(dat1-np.min(dat1))/(np.max(dat1)-np.min(dat1))
dat2 = hs[200,:]
dat2N =(dat2-np.min(dat2))/(np.max(dat2)-np.min(dat2))
ax.plot(dat1N)
ax.plot(dat2N,alpha=0.4,color='red')


# In[180]:


rastDD = rast.copy(new_array=diffDEM)
crv = xdem.terrain.curvature(rastDD)
crv.show(cmap="Greys_r",vmin=-30,vmax=30)


# In[182]:


plt.plot(crv.data[200,:])


# In[153]:


hs_sC = feature.canny(hs,sigma=4)
fig,ax = plt.subplots(1,2,figsize=(10,10))
ax[0].imshow(hs_sC,cmap='binary')

footprint = disk(4)
hs_sCo = closing(hs_sC, footprint)
ax[1].imshow(hs_sCo,cmap='binary')


# In[156]:


segment = segmentation.watershed(hs_rescale)
plt.imshow(segment,cmap='Grays')


# In[135]:


plt.plot(hs[200,:])


# In[207]:


def hillshade(array,azimuth,angle_altitude):
    azimuth = 360 - azimuth
    x, y = np.gradient(array)
    slope = np.pi/2. - np.arctan(np.sqrt(x*x + y*y))
    aspect = np.arctan2(-x, y)
    azm_rad = azimuth*np.pi/180. #azimuth in radians
    alt_rad = angle_altitude*np.pi/180. #altitude in radians
 
    shaded = np.sin(alt_rad)*np.sin(slope) + np.cos(alt_rad)*np.cos(slope)*np.cos((azm_rad - np.pi/2.) - aspect)
    
    return (255*(shaded + 1)/2, slope)


# In[ ]:


def double_gradient(array):
    x, y = np.gradient(array)
    slope = np.pi/2. - np.arctan(np.sqrt(x*x + y*y))
    x2, y2 = np.gradient(slope)
    del_slope = np.pi/2. - np.arctan(np.sqrt(x2*x2 + y2*y2))
    
    return del_slope

