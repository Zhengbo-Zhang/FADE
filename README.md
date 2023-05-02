

Our dataset, annotation instructions, dataset documentation and intended uses are published and you can download our dataset from [here](http://tuzhigang.cn/dataset/FADE.html)

![image](https://github.com/Zhengbo-Zhang/FADE/blob/880d81df366ee71df1d55ccc3473c926bd85c2b0/f7084d1276d38f271771b27442a2a80.png)

## Experiment
CPU: Intel(R) Xeon(R) Gold 5117 CPU @ 2.00GHz
GPU: GeForce RTX 3090
### BSUV-NET-2.0
source code: [BSUV-NET-2.0](https://github.com/ozantezcan/BSUV-Net-2.0)
###### Environment
1. python:	3.8.8
2. pytorch:	1.8.1
3. opencv:	4.0.1

###### Training Parameters
1. network:	 unetvgg16
2. inp_size: 600
3. empty_bg: manual
4. recent_bg: None
5. seg_ch: None
6. lr: 0.0001
7. weight_decay: 0.01
8. opt:	adam
9. aug_ioa:	None

### MOBILE-VOD
source code: [MOBILE-VOD](https://github.com/vikrant7/mobile-vod-bottleneck-lstm)
###### Environment
1. python:		3.8.12
2. pytorch:		1.11.0
3. opencv-python:	4.5.5.64

###### Training Parameters
1. network:		mvod_lstm1
2. lr:		0.003
3. momentum:	0.9
4. weight_decay:	0.0005
5. gamma:		0.1

### Opencv Background Subtract
Run `python precision_recall_tro.py --algo <k>` for `<k> = MOG2, KNN, ...` to compute the results for each fold. This code will save the results to `result.txt`.

###### Environment
1. python:		3.8.8
2. opencv-contrib-python:	4.5.5.64
##### MOG2
[opencv::createBackgroundSubtractorMOG2()](https://docs.opencv.org/3.4/d7/d7b/classcv_1_1BackgroundSubtractorMOG2.html)
1. history:	300
2. varThreshold:	100
3. detectShadow:	false

##### KNN
[opencv::createBackgroundSubtractorKNN()](https://docs.opencv.org/4.x/db/d88/classcv_1_1BackgroundSubtractorKNN.html)
1. history:	300
2. varThreshold:	100
3. detectShadow:	false

##### MOG
[opencv::bgsegm::createBackgroundSubtractorMOG()](https://docs.opencv.org/4.x/d6/da7/classcv_1_1bgsegm_1_1BackgroundSubtractorMOG.html)
- default

##### CNT
[opencv::bgsegm::createBackgroundSubtractorCNT()](https://docs.opencv.org/4.x/de/dca/classcv_1_1bgsegm_1_1BackgroundSubtractorCNT.html)
1. setIsParallel
2. setUseHistory
3. setMinPixelStability(1)
4. setMaxPixelStability(4)

##### GMG
[cv::bgsegm::BackgroundSubtractorGMG](https://docs.opencv.org/3.4/d1/d5c/classcv_1_1bgsegm_1_1BackgroundSubtractorGMG.html)
1. setNumFrames(5)
2. setUpdateBackgroundModel(True)

##### GSOC
[cv::bgsegm::BackgroundSubtractorGSOC](https://docs.opencv.org/3.4/d4/dd5/classcv_1_1bgsegm_1_1BackgroundSubtractorGSOC.html)
1. history:	300
2. varThreshold:	100

##### LSBP
[cv::bgsegm::BackgroundSubtractorLSBP](https://docs.opencv.org/3.4/de/d4c/classcv_1_1bgsegm_1_1BackgroundSubtractorLSBP.html)
- default

### VIBE
Run `python vibe.py` to compute the results for each fold. This code will save the results to `result.txt`.
1. num_sam:	20
2. min_match: 2
3. radiu: 20
4. rand_sa:	16
