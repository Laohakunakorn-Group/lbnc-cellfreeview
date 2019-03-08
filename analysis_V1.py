import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm as tqdm
import pyfiglet

import cellfreeview as cfv

# Set up files to read
FILENAME = 'compRRR4calib'
FILEPATH = './imgs/'
OUTPATH = './output/'
RINGS = [1,2,3,4,5,6,7,8]
REFIMG = 0 # Fluorescence reference image number
MODE = 1
CROPSIZEH = 400 # height ### default 400
CROPSIZEW = 280 # width ### default 280
ROI_bx=22 # bright ROI x
ROI_by=120 # bright ROI y
ROI_dx=22 # dark ROI
ROI_dy=120 # dark ROI
roicoords = cfv.calculateROI(CROPSIZEW,CROPSIZEH,[ROI_bx,ROI_by,ROI_dx,ROI_dy])

def main():

    print()
    print(pyfiglet.figlet_format("LBNC"))
    print('Welcome to the LBNC chemostat data analysis pipeline')
    print('Inputs are tiff images whose filenames are of the form "...ringindex_*_imagenumber.tiff" ')
    print()


    if MODE==1:
        print('Mode 1: reference fluor image')
    elif MODE==2:
        print('Mode 2: individual centres')
    elif MODE==3:
        print('Mode 3: edgefile')

    ### Read requested images into image collection

    data = {}
    flux = {}
    RINGSTOREAD = [item-1 for item in RINGS]

    print('Reading data...')
    for ringindex in tqdm.tqdm(RINGSTOREAD):
        ic,max_ind,min_ind,numberoffiles=cfv.readOneRing(ringindex,FILEPATH,FILENAME)
        shape = np.shape(ic[0])

        # For each ring, can get angle/centre from four sources:
        # 1. Fluorescence from reference image
        # 2. Fluorescence from each frame individually
        # 3. From edgefile
        # 4. From brightfield reference image

        angle,centre = cfv.getAnglesCentres(ic,max_ind,min_ind,numberoffiles,ringindex,REFIMG,OUTPATH,MODE)

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
            # 4. Plot single image and view ROI
            ring=input('Please select ring (1-8). Press ENTER to cancel: ')
            if ring=='':
                pass
            else:
                CYCLE=input('Please select timepoint (cycle number). Press ENTER to cancel:')
                if CYCLE=='':
                    pass
                else:
                    cfv.plotSingleImage(data,ring,int(CYCLE),OUTPATH,roicoords,save=False)

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
    print('4. Plot single image and view ROI')
    print('5. Save all plots')
    print('Press ENTER to exit without saving.')
    print()

# Call main function.

if __name__=='__main__':
    main()
