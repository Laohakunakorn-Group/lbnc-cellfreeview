# cellfreeview.py
# This file defines all functions used for the chemostat analysis code
# as called from ellfreeanalysis.py

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.ticker import AutoMinorLocator

import numpy as np
import pandas as pd
import math as math
import glob as glob
import re as re
import skimage.io as skio
from skimage.filters import threshold_otsu
from skimage import feature
from scipy import ndimage
from scipy import stats
from sklearn.cluster import KMeans
import skimage.exposure as skex
import datetime

#### Define functions for analysis

def readOneRing(ringindex,FILEPATH,FILENAME,TRANSPOSE):
    # Read in all images corresponding to a particular chemostat ring
    # Inputs are tiff images whose filenames are of the form '...ringindex_*_imagenumber.tiff'
    # where 'ringindex' and 'imagenumber' identify the ring and sequence number

    # Output is an image collection, max and min values, and the number of files
    # Output image collection shape is (sequence, x ,y)

    listfiles=glob.glob(FILEPATH+FILENAME+'_'+str(int(ringindex))+'_*.tiff')
    numberoffiles=len(listfiles)

    # Sort filenames by sequence number of images
    listfiles=sorted(listfiles,key=lambda x:float(re.findall(r'\d\d\d\d\d\.',x)[0]))

    # Transposed for new Hama camera the images must be transposed 90 degrees
    ic=np.transpose(skio.imread_collection(listfiles),axes=TRANSPOSE)

    max_ind=np.zeros(numberoffiles) # Report max value
    min_ind=np.zeros(numberoffiles) # Report min value

    for i in range(numberoffiles):
        max_ind[i]=np.max(ic[i,:,:])
        min_ind[i]=np.min(ic[i,:,:])

    return ic,max_ind,min_ind,numberoffiles,listfiles


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

    edges=feature.canny(denoise,sigma=sigma) # Find edges using skimage canny. # <<<<<<<<========

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

def getAnglesCentres(ic,max_ind,min_ind,numberoffiles,ringindex,REFIMG,OUTPATH,MODE):
    # Get angles and centres for each image according to MODE 1-4:
    # 1. Fluorescence from reference image
    # 2. Fluorescence from each frame individually
    # 3. From edgefile
    # 4. From brightfield reference image
    # Returns (angle,centre) arrays.

    angle=np.ones(numberoffiles)
    centre=np.ones(numberoffiles)

    if MODE==1:

        a,b = findEdgesFluor(ic,max_ind[REFIMG],min_ind[REFIMG],REFIMG)

        if math.isnan(a)==True:
            a=0
            print('Cannot determine angle for ring '+str(ringindex+1))
        angle = a*angle

        if math.isnan(b)==True:
            b=shape[0]/2
            print('Cannot locate centre for ring '+str(ringindex+1))
        centre = b*centre

    elif MODE==2:

        for i in range(numberoffiles):
            angle[i],centre[i] = findEdgesFluor(ic,max_ind[i],min_ind[i],i)
            if math.isnan(angle[i])==True:
                angle[i]=0
                print('Cannot determine angle for ring '+str(ringindex+1)+', frame '+str(i))

            if math.isnan(centre[i])==True:
                centre[i]=shape[0]/2
                print('Cannot locate centre for ring '+str(ringindex+1)+', frame '+str(i))

        centreframe=pd.DataFrame(columns=['angle','centre'])
        centreframe['angle']=angle
        centreframe['centre']=centre
        centreframe.to_csv(path_or_buf=OUTPATH+'centres_'+str(ringindex+1)+'.csv', sep='\t')

    elif MODE==3:

        centreframe=pd.read_csv(OUTPATH+'centres_'+str(ringindex+1)+'.csv', sep='\t')
        angle=centreframe['angle']
        centre=centreframe['centre']

    return(angle,centre)


def rotateAndCrop(numberoffiles,imagestack,CROPSIZEH,CROPSIZEW,angle,shape,centre):
    # Rotate and crop images according to ROI, angle, and centre information

    newstack=np.empty([numberoffiles,CROPSIZEH,CROPSIZEW])
    for i in range(numberoffiles):
        temp=ndimage.interpolation.rotate(imagestack[i,:,:],-angle[i],reshape=False) # How to define direction of rotation? Seems arbitrary at the moment.
        newstack[i,:,:]=temp[int(shape[1]/2)-int(CROPSIZEH/2):int(shape[1]/2)+int(CROPSIZEH/2),
                            int(centre[i]-CROPSIZEW/2):int(centre[i]+CROPSIZEW/2)]

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


#### Define functions for calibration

def wtlsq(x,y,w):
    # Weighted least squares fit coded by hand.
    # Calculates standard errors and confidence intervals using t-distribution.

    # 1. Load Data

    n=len(x)

    # 2. Fit to equation y=a+bx by direct calculation

    # 2.1 Sum of elements

    s2=np.sum(1/w**2)
    sx=np.sum(x/w**2)
    sx2=np.sum(x**2/w**2)
    sy=np.sum(y/w**2)
    sy2=np.sum(y**2/w**2)
    sxy=np.sum(x*y/w**2)
    xbar=sx/n
    ybar=sy/n

    # 2.2 Least squares calculation for a and b

    a=(sx2*sy-sx*sxy)/(s2*sx2-sx**2)
    b=(s2*sxy-sx*sy)/(s2*sx2-sx**2)
    yhat=a+b*x

    # 2.3 Sum of squared residuals

    chi2=np.sum((y-yhat)**2)

    # 2.4 Standard deviation and standard errors

    sa=sx2**0.5/(s2*sx2-sx**2)**0.5
    sb=s2**0.5/(s2*sx2-sx**2)**0.5
    s=(n/(n-2))**0.5/s2*(s2*sy2-sy**2-(s2*sxy-sx*sy)/(s2*sx2-sx**2))**0.5

    k=stats.t.ppf(1-0.05/2,n-2) # Coverage interval at 95% confidence level

    sy0=(s**2/n+(x-xbar)**2*s2/(s2*sx2-sx**2))**0.5

    yhatplus=yhat+k*sy0 # Change here for SE or confidence intervals.
    yhatminus=yhat-k*sy0

    # 2.5 Linear correlation coefficient
    r=(s2*sxy-sx*sy)/((s2*sx2-sx**2)**0.5*(s2*sy2-sy**2)**0.5)

    return(a,b,sa,sb,k*sa,k*sb,r)


def calibrateDilute(flux,offset,ringindex, fits, ratios, params):

    # Linearise data
    # We expect dilution function of the form cn/c0=(1-VD/VT)**N
    # Linerisation is therefore Z*log(f)=log(cn/c0)=N*log(1-VD/VT)

    c0=flux[offset,4] # Set initial flux for normalisation. Offset by N=1.
    deltac0=flux[offset,5]
    cn=flux[offset:,4] # flux data
    deltacn=flux[offset:,5] # flux error bars

    f=cn/c0
    deltaf=np.sqrt((deltacn/cn)**2+(deltac0/c0)**2)

    N=np.arange(len(flux[offset:,4]))
    Z=np.log(f)
    deltaZ=deltaf/f

    # Fit data to form y=a+b*x. We let a float (don't set a=0).

    (a,b,sa,sb,ksa,ksb,r)=wtlsq(N,Z,deltaZ) # Weighted least-squares fit y=a+b*x

    # Let VD/VT=dilratio

    dilratio=1-np.exp(b)
    deltadilratio=np.exp(b)*sb # Direct error
    deltakdilratio=np.exp(b)*ksb # 95% confidence error

    newX=np.linspace(0,max(N)+offset,300) # New x axis for fit plot, smoothed with 300 points.
    yhat=c0*(1-dilratio)**(newX-offset) # Offset fit by starting value.
    yhatplus=c0*(1-dilratio+deltadilratio)**(newX-offset) # Upper bound from direct error.
    yhatminus=c0*(1-dilratio-deltadilratio)**(newX-offset) # Lower bound from direct error.

    fits['Ring '+str(ringindex+1)] = [yhat,yhatplus,yhatminus]
    ratios['Ring '+str(ringindex+1)] = [dilratio, deltadilratio, deltakdilratio]
    params['Ring '+str(ringindex+1)] = [a,b,sa,sb,ksa,ksb,r]

    return(newX,fits,ratios,params)


#### Define functions for plotting

def plotInitialise(figW,figH):

    plt.close("all")
    figure_options={'figsize':(figW,figH)} # figure size in inches. A4=11.7x8.3, A5=8.3,5.8
    font_options={'size':'14','family':'sans-serif','sans-serif':'Arial'}
    plt.rc('figure', **figure_options)
    plt.rc('font', **font_options)

def plotFormat(ax,xlabel=False,
                    ylabel=False,
                    xlim=False,
                    ylim=False,
                    title=False,
                    xticks=False,
                    yticks=False,
                    logx=False,
                    logy=False,
                    logxy=False,
                    symlogx=False,
                    legend=False):

    # Set titles and labels
    if title!=False:
        ax.set_title(title)
    if xlabel!=False:
        ax.set_xlabel(xlabel, labelpad=12)
    if ylabel!=False:
        ax.set_ylabel(ylabel, labelpad=12)

    # Set axis limits
    if xlim!=False:
        ax.set_xlim(xlim)
    if ylim!=False:
        ax.set_ylim(ylim)

    # Set tick values
    if xticks!=False:
        ax.set_xticks(xticks)
    if yticks!=False:
        ax.set_yticks(yticks)

    # Set line thicknesses
    #ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%1.e"))
    #ax.axhline(linewidth=2, color='k')
    #ax.axvline(linewidth=2, color='k')
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)

    # Set ticks
    if logx==True:
        ax.set_xscale("log")

    elif logy==True:
        ax.set_yscale("log")

    elif logxy==True:
        ax.set_xscale("log")
        ax.set_yscale("log")

    elif symlogx==True:
        ax.set_xscale("symlog",linthreshx=1e-4)
        ax.set_yscale("log")

    else:
        minorLocatorx=AutoMinorLocator(2) # Number of minor intervals per major interval
        minorLocatory=AutoMinorLocator(2)
        ax.xaxis.set_minor_locator(minorLocatorx)
        ax.yaxis.set_minor_locator(minorLocatory)

    ax.tick_params(which='major', width=2, length=8, pad=9,direction='in',top='on',right='on')
    ax.tick_params(which='minor', width=2, length=4, pad=9,direction='in',top='on',right='on')

    if legend==True:
        ax.legend(loc='upper right', fontsize=14,numpoints=1) ### Default 'best'


def plotTotalImages(data,numberoffiles,RINGSTOREAD,OUTPATH,minmax,save=True):

    number_rows = len(RINGSTOREAD)
    number_cols = numberoffiles

    plotInitialise(8.3,8.3/number_cols*number_rows)

    ######### CALL PLOTS #########

    fig=plt.figure(1,(number_cols,number_rows))

    grid=ImageGrid(fig,111,nrows_ncols=(number_rows,number_cols),axes_pad=0.001)
    print('Plotting images...')
    for i in range(number_rows):
        imagesring=data['Ring '+str(RINGSTOREAD[i]+1)]
        for j in range(number_cols):
            grid[j+i*number_cols].imshow(imagesring[j,:,:]).set_clim(minmax[0]*0.9,minmax[1]*1.1)
            grid[j+i*number_cols].set_xticks([])
            grid[j+i*number_cols].set_yticks([])

    if save==True:
        filename_im=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+'_imagetotal.pdf'
        plt.savefig(OUTPATH+filename_im,dpi=150,bbox_inches='tight')
    else:
        plt.show()

def plotSingleImage(data,RING,CYCLE,OUTPATH,roicoords,minmax,save=True):

    plotInitialise(8.3,5.8)

    ######### CALL PLOTS #########

    fig=plt.figure(); ax=fig.add_subplot(1,1,1)

    imagesring=data['Ring '+str(RING)]
    ax.imshow(imagesring[CYCLE,:,:]).set_clim(minmax[0]*0.9,minmax[1]*1.1)
    ax.add_patch(patches.Rectangle((roicoords[0],roicoords[2]),
                            roicoords[1]-roicoords[0],roicoords[3]-roicoords[2],fill=False,edgecolor='red'))
    ax.add_patch(patches.Rectangle((roicoords[4],roicoords[6]),
                            roicoords[5]-roicoords[4],roicoords[7]-roicoords[6],fill=False,edgecolor='red'))

    if save==True:
        filename_im=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+'_imageROI.pdf'
        plt.savefig(OUTPATH+filename_im,dpi=150,bbox_inches='tight')
    else:
        plt.show()

def plotFluxes(flux,RINGSTOREAD,OUTPATH,save=True):

    number_rows = len(RINGSTOREAD)

    plotInitialise(8.3,5.8)

    ######### CALL PLOTS #########

    fig=plt.figure(); ax=fig.add_subplot(1,1,1)

    for i in range(number_rows):
        N=np.arange(len(flux['Ring '+str(RINGSTOREAD[i]+1)][:,4]))
        ax.errorbar(x=N,
                    y=flux['Ring '+str(RINGSTOREAD[i]+1)][:,4],
                    yerr=flux['Ring '+str(RINGSTOREAD[i]+1)][:,5],fmt='o-',label='Ring '+str(RINGSTOREAD[i]+1))

    plotFormat(ax,xlabel='Cycle',ylabel='RFU',legend=True)

    if save==True:
        filename_im=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+'_ring'+''.join([str(item+1) for item in RINGSTOREAD])+'_flux.pdf'
        plt.savefig(OUTPATH+filename_im,dpi=150,bbox_inches='tight')
    else:
        plt.show()

def plotCalib(flux,newX,fits,ringindex,OUTPATH,save=True):

    plotInitialise(8.3,5.8)

    ######### CALL PLOTS #########

    fig=plt.figure(); ax=fig.add_subplot(1,1,1)

    N=np.arange(len(flux['Ring '+str(ringindex+1)][:,4]))
    ax.errorbar(x=N,
                y=flux['Ring '+str(ringindex+1)][:,4],
                yerr=flux['Ring '+str(ringindex+1)][:,5],fmt='o',label='Ring '+str(ringindex+1))
    ax.plot(newX,fits['Ring '+str(ringindex+1)][0],'-')

    plotFormat(ax,xlabel='Cycle',ylabel='RFU',legend=True)

    if save==True:
        filename_im=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+'_ring'+str(ringindex+1)+'_calib.pdf'
        plt.savefig(OUTPATH+filename_im,dpi=150,bbox_inches='tight')
    else:
        plt.show()


#### Define logging functions

def writeLog(ringindex,OUTPATH,numberoffiles,shape,minvalue,maxvalue,listfiles,mode,angle,cropsizeh,cropsizew,roicoords):

    today=datetime.datetime.today()

    with open(OUTPATH+'log'+datetime.datetime.now().strftime("%Y%m%d")
                +'_Ring'+str(int(ringindex+1))+'.txt', 'w') as LOG:

        LOG.write('Chemostat Data Analysis')
        LOG.write('\n')
        LOG.write(str(today))
        LOG.write('\n')
        LOG.write('\n')
        LOG.write('1. Data properties')
        LOG.write('\n')
        LOG.write('Number of images: '+str(numberoffiles))
        LOG.write('\n')
        LOG.write('Dimensions: '+str(shape))
        LOG.write('\n')
        LOG.write('Intensity range: '+str(minvalue)+' to '+str(maxvalue))
        LOG.write('\n')
        LOG.write('\n')
        LOG.write('Filenames: '+str(listfiles))
        LOG.write('\n')
        LOG.write('\n')
        LOG.write('Mode: '+str(mode))
        LOG.write('\n')
        LOG.write('\n')
        LOG.write('2. Image processing steps')
        LOG.write('\n')
        for i in range(numberoffiles):
            LOG.write('Rotated by '+str('{:.3f}'.format(angle[i]))+' degrees')
            LOG.write('\n')
        LOG.write('Cropped to rectangle of size '+str(cropsizeh)+' px by '+str(cropsizew)+' px')
        LOG.write('\n')
        LOG.write('\n')
        LOG.write('3. Data analysis steps')
        LOG.write('\n')
        LOG.write('Dark subtraction using bright and dark ROIs:')
        LOG.write('\n')
        LOG.write('Bright ROI = '+str(roicoords[0])+'x'+str(roicoords[1])+' px')
        LOG.write('\n')
        LOG.write('Dark ROI = '+str(roicoords[2])+'x'+str(roicoords[3])+' px')
        LOG.write('\n')

    LOG.closed

def writeLogCalib(ringindex,OUTPATH,params,ratios,ref_img,offset):

    today=datetime.datetime.today()

    with open(OUTPATH+'log'+datetime.datetime.now().strftime("%Y%m%d")
                +'_Ring'+str(int(ringindex+1))+'.txt', 'a') as LOG:

        [a,b,sa,sb,ksa,ksb,r] = params['Ring '+str(ringindex+1)]
        [dilratio, deltadilratio, deltakdilratio] = ratios['Ring '+str(ringindex+1)]

        LOG.write('Dilution calibration results')
        LOG.write('\n')
        LOG.write('Reference image: '+str(ref_img))
        LOG.write('\n')
        LOG.write('Offset: '+str(offset))
        LOG.write('\n')
        LOG.write('\n')
        LOG.write('We expect a dilution function of cn/c0=(1-VD/VT)**N.')
        LOG.write('\n')
        LOG.write('The linearisation is therefore Z*log(f)=log(cn/c0)=N*log(1-VD/VT).')
        LOG.write('\n')
        LOG.write('\n')
        LOG.write('WLS fitting to y=a+bx yields the dilution ratio VD/VT.')
        LOG.write('\n')
        LOG.write('The values of a and b are equal to')
        LOG.write('\n')
        LOG.write('a ='+str('{:.4f}'.format(a)))
        LOG.write('\n')
        LOG.write('b ='+str('{:.4f}'.format(b)))
        LOG.write('\n')
        LOG.write('\n')
        LOG.write('The standard errors are given by')
        LOG.write('\n')
        LOG.write('sa ='+str('{:.4f}'.format(sa)))
        LOG.write('\n')
        LOG.write('sb ='+str('{:.4f}'.format(sb)))
        LOG.write('\n')
        LOG.write('\n')
        LOG.write('The 95 percent confidence boundaries are given by')
        LOG.write('\n')
        LOG.write('ksa ='+str('{:.4f}'.format(ksa)))
        LOG.write('\n')
        LOG.write('ksb ='+str('{:.4f}'.format(ksb)))
        LOG.write('\n')
        LOG.write('\n')
        LOG.write('The linear correlation coefficient of the fit is')
        LOG.write('\n')
        LOG.write('r ='+str('{:.4f}'.format(r)))
        LOG.write('\n')
        LOG.write('\n')
        LOG.write('The dilution ratio VD/VT is: '+str('{:.3f}'.format(dilratio))+' +/- '+str('{:.3f}'.format(deltadilratio))+' (+/- '+str('{:.3f}'.format(deltakdilratio))+')')
        LOG.write('\n')
        LOG.write('where the 95 percent confidence interval is given in brackets')

    LOG.closed
