python main.py --help
usage: main.py [-h] [-init {auto,manual}] [-k K] [-m M] [-s]

A program for model-based segmentation of the upper and lower incisors in
panoramic radiographs

optional arguments:
  -h, --help            show this help message and exit
  -init {auto,manual}, --init_method {auto,manual}
                        The method of finding initial estimate
  -k K, --k K           No. of pixels on either side of a model point for grey
                        level model
  -m M, --m M           No. of sample points on either side of current point
                        for search
  -s, --skip_amf        Skip adaptive median filter in Preprocessing
