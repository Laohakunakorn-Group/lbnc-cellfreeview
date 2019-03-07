import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import numpy as np
import glob as glob
import skimage.io as skio
from skimage.filters import threshold_otsu
from skimage import feature
from scipy import ndimage
from sklearn.cluster import KMeans
import skimage.exposure as skex

# Define functions for image analysis 

def readOneRing(ringindex,FILEPATH,FILENAME):
    # Read in all images corresponding to a particular chemostat ring
    # Inputs are tiff images whose filenames end in '...ringindex_*.tiff' 
    # where 'ringindex' identifies the ring

    # Output is an image collection, max and min values, and the number of files
    # Output image collection shape is (sequence, x ,y)
    
    listfiles=glob.glob(FILEPATH+FILENAME+'_'+str(int(ringindex))+'_*.tiff')    
    numberoffiles=len(listfiles)

    # Transposed for new Hama camera the images must be transposed 90 degrees 
    ic=np.transpose(skio.imread_collection(listfiles),axes=(0,2,1)) 
    
    max_ind=np.zeros(numberoffiles) # Report max value
    min_ind=np.zeros(numberoffiles) # Report min value
    
    for i in range(numberoffiles):
        max_ind[i]=np.max(ic[i,:,:])
        min_ind[i]=np.min(ic[i,:,:])
            
    return ic,max_ind,min_ind,numberoffiles


def findEdgesFluor(imagestack,max_ind,min_ind,imagenumber,sigma=10,number_of_slices=4): 
    # Determine the centre and angle of each image from analysis of fluorescence signal

    ### Threshold, binary, and denoise image.
    threshold=threshold_otsu(imagestack[int(imagenumber),:,:])

    binary=imagestack[int(imagenumber),:,:]>threshold
    binimg=np.where(binary==True,1.0,0.0)

    if max_ind>(4*min_ind): # Define low contrast images.
        denoise=ndimage.median_filter(binimg,size=2)
    else:
        denoise=ndimage.median_filter(binimg,size=6) # <<<<===== size=6 for low-constrast

    edges=feature.canny(denoise,sigma=sigma) # Find edges using skimage canny. # <<<<<<<<<<==========

    shape=np.shape(edges)
    limiter=np.zeros(number_of_slices)
    section=np.zeros(number_of_slices*int(shape[1])).reshape(number_of_slices,int(shape[1]))
    locs=np.zeros(2*number_of_slices).reshape(number_of_slices,2)
    loclist=[]

    ### Centre locator
    ycentre=shape[1]/2
    centrepeaks=np.nonzero(edges[int(ycentre),:]!=0)[0]

    # We need to ensure we have only two non-zero locations for edges:
    if len(centrepeaks)>2:
        a=np.arange(len(centrepeaks))
        b=centrepeaks
        kmeans=KMeans(n_clusters=2,n_init=1)
        kmeans.fit_predict(np.transpose(np.stack((a.T,b.T))))
        c=kmeans.cluster_centers_
        centrepeaks=np.sort(c[:,1])
    else:
        centrepeaks=centrepeaks
    centre=np.average(centrepeaks)


    ### Angle calculation

    for i in range(number_of_slices):
        limiter[i]=int(shape[1]*(i+1)/(number_of_slices+1))
        section[i]=edges[int(limiter[i]),:]
        loclist.append(np.nonzero(section[i]!=0)[0])

        # We need to ensure we have only two non-zero locations for edges:
        if len(loclist[i])>2:
            a=np.arange(len(loclist[i]))
            b=loclist[i]
            kmeans=KMeans(n_clusters=2,n_init=1)
            kmeans.fit_predict(np.transpose(np.stack((a.T,b.T)))) # Reduce edge locations to 2.
            c=kmeans.cluster_centers_
            locs[i]=c[:,1]
        elif len(loclist[i])<2:
            break
        else:
            locs[i]=loclist[i]

    angle1=np.zeros(number_of_slices-1)
    angle2=np.zeros(number_of_slices-1)

    for i in range(number_of_slices-1):
        dx1=locs[i+1,0]-locs[i,0]
        dx2=locs[i+1,1]-locs[i,1]
        dy=shape[0]/(number_of_slices+1)
        angle1[i]=(np.arctan(dx1/dy))/(2*np.pi)*360 # Calculate angle from vertical.
        angle2[i]=(np.arctan(dx2/dy))/(2*np.pi)*360 # Calculate angle from vertical.
    angles=np.concatenate((angle1,angle2))
    angle_avg=np.average(angles)

    return(angle_avg,centre)

def rotateAndCrop(numberoffiles,imagestack,CROPSIZEH,CROPSIZEW,angle,shape,centre):
    # Rotate and crop images according to ROI, angle, and centre information

    newstack=np.empty([numberoffiles,CROPSIZEH,CROPSIZEW])
    for i in range(numberoffiles):
        temp=ndimage.interpolation.rotate(imagestack[i,:,:],-angle[i],reshape=False) # How to define direction of rotation? Seems arbitrary at the moment.
        newstack[i,:,:]=temp[int(shape[1]/2)-int(CROPSIZEH/2):int(shape[1]/2)+int(CROPSIZEH/2),int(centre[i]-CROPSIZEW/2):int(centre[i]+CROPSIZEW/2)]

    return(newstack)

def calculateROI(CROPSIZEW,CROPSIZEH,roi):

    ROI_bx,ROI_by,ROI_dx,ROI_dy = roi

    x_b1=int(CROPSIZEW/2-ROI_bx/2) # Light crop is centered
    x_b2=int(CROPSIZEW/2+ROI_bx/2)
    y_b1=int(CROPSIZEH/2-ROI_by/2)
    y_b2=int(CROPSIZEH/2+ROI_by/2)

    x_d1=int(0.1*CROPSIZEW-ROI_dx/2) # Location of dark crop
    x_d2=int(0.1*CROPSIZEW+ROI_dx/2)
    y_d1=int(CROPSIZEH/2-ROI_dy/2)
    y_d2=int(CROPSIZEH/2+ROI_dy/2)

    return([x_b1,x_b2,y_b1,y_b2,x_d1,x_d2,y_d1,y_d2])

def calculateFluxes(newstack,numberoffiles,roicoords):
    # Calculate fluxes from image according to ROI

    x_b1,x_b2,y_b1,y_b2,x_d1,x_d2,y_d1,y_d2 = roicoords

    # 4.4 Calculate fluxes

    flux=np.zeros(numberoffiles*6).reshape(numberoffiles,6)

    for i in range(numberoffiles):
        flux[i,0]=np.average(newstack[i,y_b1:y_b2,x_b1:x_b2]) # average bright ##### NB images are rotated 90 degrees.
        flux[i,1]=np.std(newstack[i,y_b1:y_b2,x_b1:x_b2]) # std bright
        flux[i,2]=np.average(newstack[i,y_d1:y_d2,x_d1:x_d2]) # average dark
        flux[i,3]=np.std(newstack[i,y_d1:y_d2,x_d1:x_d2]) # std dark
        flux[i,4]=flux[i,0]-flux[i,2] # dark-corrected fluorescence
        flux[i,5]=np.sqrt(np.square(flux[i,1])+np.square(flux[i,3])) # Combined errors in quadrature

    return(flux)



def plotTotalImages(data,numberoffiles,RINGSTOREAD):

    number_rows = len(RINGSTOREAD)
    number_cols = numberoffiles

    plt.close("all")

    my_dpi=150

    figure_options={'figsize':(11.7,11.7/number_cols*number_rows)} #figure size in inches. A4=11.7x8.3. 
    font_options={'size':'14','family':'sans-serif','sans-serif':'Arial'}
    plt.rc('figure', **figure_options)
    plt.rc('font', **font_options)

    ######### CALL PLOTS #########

    fig=plt.figure(1,(number_cols,number_rows))

    grid=ImageGrid(fig,111,nrows_ncols=(number_rows,number_cols),axes_pad=0.001)
    for i in range(number_rows):
        imagesring=data['Ring '+str(RINGSTOREAD[i]+1)]
        for j in range(number_cols):
            grid[j+i*number_cols].imshow(imagesring[j,:,:]).set_clim(0,4000)
            grid[j+i*number_cols].set_xticks([])
            grid[j+i*number_cols].set_yticks([])

    plt.show()

