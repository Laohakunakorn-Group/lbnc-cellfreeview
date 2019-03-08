# lbnc-cellfreeview
View and analyse data from the cell-free chemostats at [LBNC](http://lbnc.epfl.ch/).

<img src="/fig/fig.png" width="640" alt="Cell-free chemostats" align="center" hspace="40" vspace="15">

This script analyses sequential timelapse images. The signal appears as fluorescence within a microfluidic channel, which is used to rotate and centre the image stack. ROIs are defined in the bright and dark regions, and intensities averaged within them. The script produces timeseries data of fluorescence intensity for each microchemostat reactor as well as merged images and plots. Analysis of a calibration experiment determines the dilution rate within each reactor. This code is used to analyse experiments run on chips as described in [Niederholtmeyer et al. 2013](https://www.pnas.org/content/110/40/15985) and [Niederholtmeyer et al. 2015](https://elifesciences.org/articles/09771).  

To install dependencies, run

	pip install -r requirements.txt

There are three files:

* `cellfreeanalysis.py`, the main script
* `functions.py`, which contains all the function definitions required by the main script
* `config.py`, which is used to set analysis parameters

Running the code generates `*.csv` output data files, `*.pdf` plots and images, and `*.txt` log files. These are directed to an output directory as defined within the config file. 

