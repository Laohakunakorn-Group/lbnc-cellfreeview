# config.py

# Set filename and input and output paths.
# Inputs are tiff images whose filenames are of the form
# '..._ringindex_*_imagenumber.tiff'
# ringindex is first number to parse, 
# imagenumber is final number to parse.
FILENAME = 'compRRR4calib' 
FILEPATH = './imgs/' # input files
OUTPATH = './output/'
EDGEFILE = 'centresGLOB.csv' # comma-delimited edgefile

# Rings to analyse, from 1-8.
RINGS = [1,2,3,4,5,6,7,8]

# For each ring, can get angle/centre from four sources:
# 1. Fluorescence from reference image
# 2. Fluorescence from each frame individually
# 3. From timeseries edgefile
# 4. From global edgefile 
# 5. From brightfield reference image (not yet implemented)
MODE = 1

# Fluorescence reference image. The number corresponds to the
# sequence number of the image, beginning at 0. Choose an image with large signal.
REFIMG = 0

# Index to start fitting from for calibration.
OFFSET = 1

# Default image size is 512x512 but this may change due to binning.
# Parameters for rotation, cropping, and ROI can be adjusted here accordingly.
TRANSPOSE = (0,2,1) # swap the 2nd and 3rd indices for 90-degree rotation
CROPSIZEH = 400 # height ### default 400
CROPSIZEW = 280 # width ### default 280
ROI_bx = 22 # bright ROI x
ROI_by = 120 # bright ROI y
ROI_dx = 22 # dark ROI
ROI_dy = 120 # dark ROI




