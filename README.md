# The Flag Median and FlagIRLS

The python code to reproduce results for the CVPR 2022 paper by Nathan Mankovich et. al.

## Description

CVPR_Examples is the catch all example notebook to re-run all examples from the CVPR 2022 paper The Flag Median and FlagIRLS. The models directory contains a pretrained pytorch model on MNIST. For preprocessing and origin of the subspace MindsEye data see [Marrinan et. al.](www.cs.colostate.edu/~vision/summet). For the origin of the YouTube LBG data see the YouTube Action Data Set [Liu et. al.](https://www.crcv.ucf.edu/data/UCF_YouTube_Action.php). Most of the functions for the examples are in center_algorithms.py. run_lbg_minds_eye.py and youtube_lbg.py are scripts to run the MindsEye and YouTube subspace LBG examples. youtube_dataset.py creates a subspace dataset from the YouTube videos. Warning- the MindsEye and YouTube examples take a while to run.

### How to Cite
If you find this code useful, please consider citing us at
```
@inproceedings{mankovich2022flag,
  title={The Flag Median and FlagIRLS},
  author={Mankovich, Nathan and King, Emily J and Peterson, Chris and Kirby, Michael},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10339--10347},
  year={2022}
}
```

### Data Sets

Data sets must be downloaded seperately from the git code. All datasets go in the data directory.

## Minds Eye
* Download the data from [summet](www.cs.colostate.edu/~vision/summet)
* Save the .mat files in a data directory.

## YouTube
* Download the files from the YouTube Action Data Set on the website [youtube](https://www.crcv.ucf.edu/data/UCF_YouTube_Action.php).
* Store the action_youtube_naudio file in the data directory. 
* Make a directory in the data directory called action_youtube_gr
* Use the youtube_dataset.py to generate the subspace YouTube dataset


### Dependencies

* numpy
* matplotlib
* torch
* pandas
* seaborn
* sklearn
* time
* os
* cv2
* mat73
* scipy


## Help

None yet.

## Authors

Nathan Mankovich

## License

None Yet

## Acknowledgments

* Deng et. al. (The mnist database of handwritten digit images for machine learning research)
* Tim Marinnan et. al. (Subspace Mean and Median Evaluation Toolkit (SuMMET))
* Liu et. al. (Recognizing Realistic Actions from Videos "in the Wild").
