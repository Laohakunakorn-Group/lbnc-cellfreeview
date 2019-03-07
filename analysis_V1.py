import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import skimage.io as skio
import pandas as pd

import cellfreeview as cfv

# Set up files to read
FILENAME = 'compRRR4calib'
FILEPATH = './imgs/'
OUTPATH = './output/'
CROPSIZEH = 400 # height ### default 400
CROPSIZEW = 280 # width ### default 280
RINGSTOREAD = [0,1,2,3,4,5,6,7]
REFERENCECYCLE = 0
ROI_bx=22 # bright ROI x 
ROI_by=120 # bright ROI y
ROI_dx=22 # dark ROI
ROI_dy=120 # dark ROI
roicoords = cfv.calculateROI(CROPSIZEW,CROPSIZEH,[ROI_bx,ROI_by,ROI_dx,ROI_dy])

def main():

    print()
    print('Welcome to the chemostat data analysis pipeline')
    print('Inputs are tiff images whose filenames are of the form "...ringindex_*_imagenumber.tiff" ')
    print()

    ### Read requested images into image collection

    data = {}
    flux = {}
    for ringindex in RINGSTOREAD:
        print('Reading ring '+str(ringindex+1))
        ic,max_ind,min_ind,numberoffiles=cfv.readOneRing(ringindex,FILEPATH,FILENAME)
        
        REFERENCECYCLE = 0
        a,b = cfv.findEdgesFluor(ic,max_ind[REFERENCECYCLE],min_ind[REFERENCECYCLE],REFERENCECYCLE)
        angle = a*np.ones(numberoffiles)
        centre = b*np.ones(numberoffiles)
        shape = np.shape(ic[0])

        data['Ring '+str(ringindex+1)] = cfv.rotateAndCrop(numberoffiles,ic,CROPSIZEH,CROPSIZEW,angle,shape,centre)
        flux['Ring '+str(ringindex+1)] = cfv.calculateFluxes(data['Ring '+str(ringindex+1)],numberoffiles,roicoords)

        dataframe=pd.DataFrame(flux['Ring '+str(ringindex+1)],
                                        columns=['Avg bright',
                                                 'Stdev bright',
                                                 'Avg dark', 
                                                 'Stdev dark', 
                                                 'Avg dark-corrected intensity', 
                                                 'Error dark-corrected'])
        dataframe.to_csv(path_or_buf=OUTPATH+'flux_ring'+str(ringindex+1)+'.csv', sep='\t', index_label='Cycle')

    ### Interactive main menu

    while True:
        main_menu()

        commands_str = input('Input: ')

        if commands_str=='1':
            # 1. Plot intensity-time graph for individual ring
            ring=input('Please select ring (1-8). Press ENTER to cancel: ')
            if ring=='':
                pass
            else:
                cfv.plotFluxes(flux,[int(ring)-1],OUTPATH,save=False)

        elif commands_str=='2':
            # 2. Plot combined intensity-time graph for all rings.
                cfv.plotFluxes(flux,RINGSTOREAD,OUTPATH,save=False)

        elif commands_str=='3':
            # 3. Plot all images
            cfv.plotTotalImages(data,numberoffiles,RINGSTOREAD,OUTPATH,save=False)

        elif commands_str=='4':
            # 4. Plot single image
            ring=input('Please select ring (1-8). Press ENTER to cancel: ')
            if ring=='':
                pass
            else:
                CYCLE=input('Please select timepoint (cycle number). Press ENTER to cancel:')
                if CYCLE=='':
                    pass
                else:
                    cfv.plotSingleImage(data,ring,int(CYCLE),OUTPATH,save=False)

        elif commands_str=='5':
            # 5. Save all plots
            for ringindex in RINGSTOREAD:
                cfv.plotFluxes(flux,[ringindex],OUTPATH,save=True)
            cfv.plotFluxes(flux,RINGSTOREAD,OUTPATH,save=True)
            cfv.plotTotalImages(data,numberoffiles,RINGSTOREAD,OUTPATH,save=True)

        elif commands_str=='':
            # Press ENTER to exit without saving
            print()
            print('Goodbye!')
            print()
            break


def main_menu():
    print()
    print('Please select from the following options by typing in the number, followed by ENTER: ')
    print()
    print('1. Plot intensity-time graph for individual ring')
    print('2. Plot total intensity-time graph')
    print('3. Plot all images')
    print('4. Plot single image')
    print('5. Save all plots')
    print('Press ENTER to exit without saving.')
    print()

# Call main function.

if __name__=='__main__':
    main()  