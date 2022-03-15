# The Flag Median and FlagIRLS

The python code to reproduce results for the CVPR 2022 paperr by Nathan Mankovich et. al.

## Description

CVPR_Examples is the catch all example notebook to re-run all examples from the CVPR 2022 paper The Flag Median and FlagIRLS. The directory, data, contains all the necessary files for the MindsEye and YouTube experiments. For preprocessing and origin of the subspace MindsEye data see [Marrinan et. al.](www.cs.colostate.edu/~vision/summet). For the origin of the YouTube LBG data see the YouTube Action Data Set [Liu et. al.](https://www.crcv.ucf.edu/data/UCF_YouTube_Action.php). Most of the functions for the examples are in center_algorithms.py. run_lbg_minds_eye.py and youtube_lbg.py are scripts to run the MindsEye and YouTube subspace LBG examples. youtube_dataset.py creates a subspace dataset from the YouTube videos. Warning- the MindsEye and YouTube examples take a while to run.


### Dependencies

* numpy
* matplotlib
* torch
* pandas
* seaborn
* sklearn
* time
* Describe any prerequisites, libraries, OS version, etc., needed before installing program.
* ex. Windows 10


## Help

None yet.

## Authors

Nathan Mankovich

## License

None Yet

## Acknowledgments

Tim Marinnan et. al. (Subspace Mean and Median Evaluation Toolkit (SuMMET)) and Liu et. al. (Recognizing Realistic Actions from Videos "in the Wild").