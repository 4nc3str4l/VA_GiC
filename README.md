Background Subtraction applied to Object Recognition
======

Visió Artificial: Guillem Pascual i Cristian Muriel

Getting pre-trained cnn
------

Our pre-trained caffemodel might be downloaded from, by iterations number:

* 5000: https://www.dropbox.com/s/8mcnj8j5hhytn6z/bsaor_iter_5000.caffemodel?dl=0
* 10000: https://www.dropbox.com/s/yw5535ckknlo17d/bsaor_iter_10000.caffemodel?dl=0
* 15000: https://www.dropbox.com/s/g5gh7lmab31d0x5/bsaor_iter_15000.caffemodel?dl=0
* 20000: https://www.dropbox.com/s/vuxo1ayux971z84/bsaor_iter_20000.caffemodel?dl=0 
* 25000: https://www.dropbox.com/s/y81588ziy4iglyj/bsaor_iter_25000.caffemodel?dl=0

And be freely used. We would, however, appreciate being cited, as does caffe.

Requierements
------

You must have python (2.7 is strongly recommended, might work on 3.0), alongside with latest OpenCV distribution for python, numpy and Caffe.

Installing Anaconda Python might be the fastest way to obtain all python related requierements. OpenCV might be installed by issuing:
> conda install binstar
>
> binstar search opencv
>
> binstar show menpo/opencv
>
>conda install --channel https://conda.binstar.org/menpo opencv

Citing
====
We have used caffe (http://caffe.berkeleyvision.org/) as CNN:

> Author: Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor

> Journal: arXiv preprint arXiv:1408.5093

> Title: Caffe: Convolutional Architecture for Fast Feature Embedding

> Year: 2014
