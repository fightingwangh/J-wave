|Research Title|Research Purpose|
|---|---|
|Research on J Wave Detection Based on Transfer Learning and VGG16|This study proposes a J wave detection method based on transfer learning with a pre-trained VGG16 model. The goal is to achieve outstanding detection results on a limited J wave dataset through transfer learning.|

## J wave simulation
The "J wave simulation" folder mainly contains programs for simulating J waves. We used the e0107 dataset as an example to demonstrate the simulation process of J waves.

### e0107.mat e0107_.mat
The file e0107.mat is derived from the original e0107.dat data, while e0107_.mat is derived from the original e0107.atr data, enabling further operations in Matlab.

### find_inflection_point.m find_max_differ.m
The files find_inflection_point.m and find_max_differ.m contain the core code used to find the inflection point positions for each heartbeat.

### j_wave.m
The j_wave.m file is used to generate J waves that meet specific requirements. This portion of the code can be modified based on different amplitude and duration requirements for the J wave.

### main.m
Running main.m will generate the simulated J wave signals.

## Jwavetest
The folder "Jwavetest" mainly categorizes simulated J-wave signals and non-J-wave signals.

### cwt.py
The main function of cwt.py is to utilize continuous wavelet transformation to convert one-dimensional electrocardiogram signals into two-dimensional grayscale images.

### data partitioning.py
The main function of data_partitioning.py is to randomly partition a dataset according to a set ratio.

### vgg16.py
The main function of vgg16.py is to utilize transfer learning for classifying J-wave and non-J-wave signals.

### Ten-foldtest.py
The main function of Ten-foldtest.py is to conduct ten-fold cross-validation experiments.

### data
The main purpose of the data folder is to store experimental data, where 0 represents J-wave data and 1 represents non-J-wave data.


