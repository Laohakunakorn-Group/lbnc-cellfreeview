# cellfreeanalysis.py
# Runs chemostat analysis pipeline

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm as tqdm
import pyfiglet
import os
import sys
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

# Import functions and configuration parameters
try:
    import functions as cfv
except:
    print('Function file cellfreeview.py not found, or import errors: please check all dependencies are installed.')
    sys.exit()
try:
    from config import *
except:
    print('No config file! Please make one and re-run. Examples can be found on the GitHub repo.')
    sys.exit()

if not os.path.exists(OUTPATH):
    os.mkdir('./output/')
roicoords = cfv.calculateROI(CROPSIZEW,CROPSIZEH,[ROI_bx,ROI_by,ROI_dx,ROI_dy])


def main():

    print()
    print(pyfiglet.figlet_format("LBNC - EPFL"))
    print('Welcome to the LBNC chemostat data analysis pipeline V1.1.0, 2019')
    print('Make sure you have set required parameters in the config file')
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
    minmax = {}
    mini = [] # Keep track of global min and max values;
    maxi = [] # in later version should make this more elegant
    RINGSTOREAD = [item-1 for item in RINGS]

    print('Reading data and saving...')
    for ringindex in tqdm.tqdm(RINGSTOREAD):
        ic,max_ind,min_ind,numberoffiles,listfiles=cfv.readOneRing(ringindex,FILEPATH,FILENAME,TRANSPOSE)
        shape = np.shape(ic[0])

        angle,centre = cfv.getAnglesCentres(ic,max_ind,min_ind,numberoffiles,ringindex,shape,REFIMG,FILEPATH,OUTPATH,EDGEFILE,MODE)
        data['Ring '+str(ringindex+1)] = cfv.rotateAndCrop(numberoffiles,ic,CROPSIZEH,CROPSIZEW,angle,shape,centre)
        flux['Ring '+str(ringindex+1)] = cfv.calculateFluxes(data['Ring '+str(ringindex+1)],numberoffiles,roicoords)

        minmax['Ring '+str(ringindex+1)]=np.stack([min_ind,max_ind], axis=-1)
        mini.append(np.min(min_ind)); maxi.append(np.max(max_ind)) # Global min/max values
        dataframe=pd.DataFrame(flux['Ring '+str(ringindex+1)],
                                        columns=['Avg bright',
                                                 'Stdev bright',
                                                 'Avg dark',
                                                 'Stdev dark',
                                                 'Avg dark-corrected intensity',
                                                 'Error dark-corrected'])
        dataframe.to_csv(path_or_buf=OUTPATH+'flux_ring'+str(ringindex+1)+'.csv', sep='\t', index_label='Cycle')
        cfv.writeLog(ringindex,OUTPATH,numberoffiles,shape,min(mini),max(maxi),listfiles,MODE,angle,CROPSIZEH,CROPSIZEW,roicoords)

    ### Interactive main menu

    while True:
        main_menu()

        commands_str = input('Input: ')

        if commands_str=='1':
            # 1. Calibrate dilution

            fits = {}
            ratios = {}
            params = {}

            print('Calibrating dilution...')
            for ringindex in tqdm.tqdm(RINGSTOREAD):
                (newX,fits,ratios,params)=cfv.calibrateDilute(flux['Ring '+str(ringindex+1)],OFFSET,ringindex,fits,ratios,params)
                cfv.plotCalib(flux,newX,fits,ringindex,OUTPATH,save=True)
                cfv.writeLogCalib(ringindex,OUTPATH,params,ratios,REFIMG,OFFSET)
            with open(OUTPATH+'log'+'totalcalib.txt', 'w') as LOG:
                LOG.write('Total dilution ratios')
                LOG.write('\n')
                for ringindex in RINGSTOREAD:
                    LOG.write('Ring '+str(ringindex+1)+': '+str('{:.4f}'.format(ratios['Ring '+str(ringindex+1)][0])))
                    LOG.write('\n')
                LOG.closed

        elif commands_str=='2':
            # 2. Save all plots
            for ringindex in RINGSTOREAD:
                cfv.plotFluxes(flux,[ringindex],OUTPATH,save=True)
            cfv.plotFluxes(flux,RINGSTOREAD,OUTPATH,save=True)
            cfv.plotTotalImages(data,numberoffiles,RINGSTOREAD,OUTPATH,[min(mini),max(maxi)],save=True)
            # Plot example ROI from first ring in list, first image:
            cfv.plotSingleImage(data,(RINGSTOREAD[0]+1),0,OUTPATH,roicoords,[min(mini),max(maxi)],save=True)

        elif commands_str=='3':
            # 3. Plot intensity-time graph for individual ring
            ring=input('Please select ring (1-8). Press ENTER to cancel: ')
            if ring=='':
                pass
            else:
                cfv.plotFluxes(flux,[int(ring)-1],OUTPATH,save=False)

        elif commands_str=='4':
            # 4. Plot combined intensity-time graph for all rings.
                cfv.plotFluxes(flux,RINGSTOREAD,OUTPATH,save=False)

        elif commands_str=='5':
            # 5. Plot all images
            cfv.plotTotalImages(data,numberoffiles,RINGSTOREAD,OUTPATH,[min(mini),max(maxi)],save=False)

        elif commands_str=='6':
            # 6. Plot single image and view ROI
            ring=input('Please select ring (1-8). Press ENTER to cancel: ')
            if ring=='':
                pass
            else:
                CYCLE=input('Please select timepoint (cycle number). Press ENTER to cancel:')
                if CYCLE=='':
                    pass
                else:
                    CYCLE = int(CYCLE)
                    cfv.plotSingleImage(data,ring,CYCLE,OUTPATH,roicoords,minmax['Ring '+str(ring)][CYCLE],save=False)

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
    print('1. Calibrate dilution')
    print('2. Save all plots')
    print('3. Plot intensity-time graph for individual ring')
    print('4. Plot total intensity-time graph')
    print('5. Plot all images')
    print('6. Plot single image and view ROI')
    print('Press ENTER to exit without saving.')
    print()

# Call main function.

if __name__=='__main__':
    main()
