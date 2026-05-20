So this is the repo that stores the major work done by Sebastian Cocks for his Master Of Philosophy through Defense Trailblazer. 

The folder that contains the main code is `/mfd-net/` for the big sequence level classifier using power and phase-edge spectrograms, image encoders and a decoder.

The other folder with the single frame classifiers is the `image-classification-clean` one. This stores the 5 algorithms used to benchmark the AIMC-Spec data set. There is another directory called `image-classification` which stores (messily) the older work that didn't make it into the GitHub when publishing. On that note there is a GitHub repo at [AIMC-image-classification](https://github.com/seb-cocks/AIMC-image-classification) which is associated with the Arxiv paper of AIMC-Spec-v2. 

Now the way to use the pipeline is relatively simple. The steps are:

1. Now head into the `mfd-net` folder, theres others but this is the main one. Make sure to open the `main.py` since thats how you tell the code what driver to use. 
2. Make sure that the driver associated with the code stored inside `mfd_stream_existing_iq` is uncommented. 
    1. From the `mfd_stream_existing_iq` folder head into it's `driver.py` and change the fields that need to be change for your data. 
    2. This normally consists of the amount of windows you want and the paths of the data. 
3. In order to run the code you need to use the command of `python -m src.main` or the equivalent. This makes the code run from the `main.py` within `/src/` and makes all the relative paths for the imports work. You can also specify if the data should be usable with the single frame image encoders. Which you will need to have so that MFD-Net can run. 
4. It will generate a data set to use with the code stored in `mfd_stream_train` and `ResNet`
5. For this part it will be relatively straight forward. There are 2 main files that you need, those being `ResNet/ResNet_MemMap.py` and `mfd_stream_train/driver.py`. The one that is gonna be used first is the resnet one. 
6. Within the ResNet pyhton file you will see a bunch of configurable fields. Namely the paths for the data and the type of resnet that you want to train, i.e `dual | power | phase`. Basically just fill out the fields and the paths and then head back to the `main.py` and make sure that the correct function is uncommented and run the command `python -m src.main`. 
7. Assuming that you trained up the image encoder(s) that you wanted to use, you can head into the `mfd_stream_train/driver.py` and repeat the process of setting paramters that you did for the ResNet file. Though the parameters of note in this file are the ones that dictate how the temporal buffer works, this is set through `CHUNKY` and some others to determine stride length. Oh you also need to set the type of spectrogram that is being used and also the correct paths for the checkpoints of the image encoders that you want to use. 
8. Once all of this is set just hit go with the command of `python -m src.main`, making sure that the correct driver function is uncommented within `main.py`.
9. This will then train up the algorithm. 

Note: I (seb) didn't end up adding a way to load in the data seperately to the training pipeline but you could just cut out the testing segment of the training pipieline. 