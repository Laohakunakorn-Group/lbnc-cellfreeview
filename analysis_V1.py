import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import skimage.io as skio

import cellfreeview as cfv

# Set up files to read
FILENAME = 'compRRR4calib'
FILEPATH = './imgs/'
CROPSIZEH = 400 # height ### default 400
CROPSIZEW = 280 # width ### default 280
RINGSTOREAD = [0,1,2,3,4,5,6,7]
ROI_bx=22 # bright ROI x 
ROI_by=120 # bright ROI y
ROI_dx=22 # dark ROI
ROI_dy=120 # dark ROI
roicoords = cfv.calculateROI(CROPSIZEW,CROPSIZEH,[ROI_bx,ROI_by,ROI_dx,ROI_dy])

def main():
    # Interactive main menu

    print()
    print('Welcome to the chemostat data analysis pipeline')
    print('Input files required are of the form filename_*_...tiff where * is the ring number.')
    print()

    # Read requested images into image collection

    data = {}
    flux = {}
    for ringindex in RINGSTOREAD:
        print('Reading ring '+str(ringindex+1))
        ic,max_ind,min_ind,numberoffiles=cfv.readOneRing(ringindex,FILEPATH,FILENAME)
        
        refindex = 0
        a,b = cfv.findEdgesFluor(ic,max_ind[refindex],min_ind[refindex],refindex)
        angle = a*np.ones(numberoffiles)
        centre = b*np.ones(numberoffiles)
        shape = np.shape(ic[0])

        data['Ring '+str(ringindex+1)] = cfv.rotateAndCrop(numberoffiles,ic,CROPSIZEH,CROPSIZEW,angle,shape,centre)
        flux['Ring '+str(ringindex+1)] = cfv.calculateFluxes(data['Ring '+str(ringindex+1)],numberoffiles,roicoords)

    while True:
        main_menu()

        commands_str = input('Input: ')

        if commands_str=='1':

            cfv.plotTotalImages(data,numberoffiles,RINGSTOREAD)

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
    print('1. run command')
    print('Press ENTER to exit without saving.')
    print()

# Call main function.

if __name__=='__main__':
    main()  