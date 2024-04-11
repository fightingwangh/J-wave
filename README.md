|Research Title|Research Purpose|
|---|---|
|Research on J Wave Detection Based on Transfer Learning and VGG16|This study proposes a J wave detection method based on transfer learning with a pre-trained VGG16 model. The goal is to achieve outstanding detection results on a limited J wave dataset through transfer learning.|

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


