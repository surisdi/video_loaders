# Video Loaders

Access patterns matter a lot when choosing the most appropriate library for video loading. In this repository, we implement different video loaders (see [this list](#implemented-loaders)). We implement different access patterns, specifically, 1- loading the whole video, 2- loading random segments (clips) of the video, and 3- loading random frames. 

We also contemplate other axis of variation, such as the sampling rate, different augmentations, as well as parameters related to these (cropping size, number of frames in a segment, etc.). All of these may influence which video loaders are more efficient. Specifically, we give options to center crop, resize, and normalize according to pre-defined mean and variances. While many other possible transformations are possible, these are some standard ones that are good enough to benchmark the video loaders.

Additionally, we also implement audio loading for those video loaders that support it.

Other than being a tool for comparing loading times, this repository is also convenient to compare other aspects of the video loaders (such as the different ways they deal with different frame rates), as well as to have a working implementation of multiple video loaders that use the same data structure and parameters.



See other [Considerations](#considerations) later in this document.

[Contributions](#contributions-to-the-project) are welcome!


## Run

First, prepare the data running `preprocess.py`. Follow the instructions in that file to store the data in the correct 
format.

Then, modify `parameters.py` and run `main.py` without any arguments (the arguments are loaded from `parameters.py`).
Consider modifying the reporting/visualizations code in `main.py` to report the numbers you are interested in.


## Installation


Installation of each one of all of the implemented libraries at the same time is not possible, as they have conflicting
requirements. This is why we do not provide a single `requirements.txt` file. Install the libraries that you are most 
interested in using. Python 3.9 is the most compatible Python version for most of the libraries.

Make sure [FFmpeg](https://ffmpeg.org/download.html) is installed in your machine.

The libraries that can be installed without incompatibilities are:

```bash
pip install git+https://github.com/mondeja/moviepy.git  # Last pip package is not up to date. Install from git directly. It says it is not compatible with last numpy, but it works (do not downgrade numpy)

pip3 install torch torchvision torchaudio  # Default version. See later for alternatives.

git clone https://github.com/pytorch/vision.git
cd vision
python setup.py develop

pip install -U openmim
mim install mmcv  # Requires previous installation of torch

pip install git+https://github.com/facebookresearch/pytorchvideo.git  # Last pip package is not up to date with latest torchvision release, so install from git directly

# pip install av  # This should get installed with pytorchvideo

pip install numpy
pip install absl-py
pip install pandas
pip install tensorflow
pip install ffmpeg-python
pip install pims

conda install -c conda-forge gulpio

pip install git+https://github.com/willprice/torchvideo.git@master

pip install nvidia-dali-cuda110 
conda install -c conda-forge lintel
```

The rest of libraries can present some incompatibilities with each other. Consider the following when installing them:
- To install other versions of PyTorch (for different CUDA versions, for example), see [this link](https://pytorch.org/get-started/locally/).
- The decord library can be installed with `pip install decord` if CPU support is enough. For GPU support, install from
source following [these instructions](https://github.com/dmlc/decord?tab=readme-ov-file#install-from-source).
- Pillow-SIMD is a fast replacement for Pillow. Install either one or the other. For the standard Pillow, just run `pip install pillow`, and for Pillow-SIMD, run `CC="cc -mavx2" pip install -U --force-reinstall pillow-simd`, following [these instructions](https://github.com/uploadcare/pillow-simd?tab=readme-ov-file#installation).
- The `TorchVideoVideo` video loader requires the `lintel` library, that only work on python up to 3.9. If you want to use Python 3.10+, do not use `TorchVideoVideo`.

## Considerations

- Some of these video-loading libraries provide other functionality that may be very convenient depending on the use-case, which we may not be benchmarking in this project. This project is not meant to provide a comprehensive comparison between the video loaders. Similarly, some libraries may not benefit from the specific steps and standardization measures followed in this repository. We do not claim this is the intended or ideal loading procedure for all loaders. 
- NVVL (PyTorch wrapper [in this link](https://github.com/NVIDIA/nvvl/tree/master/pytorch)) is no longer maintained and instead is part of DALI now. Because of this, we only implement DALI, and not NVVL.
- We explicitly separated the `Pillow` and `PillowSIMD` as two different video loaders, although the implementation is the same and only one of them can be installed at a time. This is to make the choice between them explicit.
  - Other loaders that use PIL may also benefit from Pillow-SIMD instead of Pillow, and this is not explicitly contemplated in this repository. Take into account in case you compare the two versions.
- If the video file is slightly corrupt or there was a bad conversion, the loaders may think the video is longer that what it actually is, which may cause some problems in the code. We controlled for this in some places but not all.
- We only implement video loaders that work for PyTorch. Therefore JAX or Tensorflow-specific loaders 
(such as [DMVR](https://github.com/google-deepmind/dmvr)) are not implemented.
- `lintel` fails for some videos. `lintel` is used in the `TorchVideoVideo` loader. The code does not raise an error, it simply crashes with `Segmentation fault (core dumped)`. It is a similar issue to the one raised 
[here](https://github.com/dukebw/lintel/issues/14), but also for .mp4 videos, not only .webm videos. The (not ideal) fix in that case is to re-encode the video using `ffmpeg`.
- GulpIO, used in the `TorchVideoGULP` loader, does not always work properly. We noticed some intercalation of frames in some of the gulp files created during pre-processing.
- The resulting frames obtained from the different video loaders would ideally be exactly the same. However, this will not be the case. There are factors that make them return slightly different results. Some of these factors are:
  - Different resize interpolation algorithms. This could also be standardized, but we chose to keep the default ones.
  - Different ways of sampling temporally.
  - Some loaders specify start/end of the clips with seconds, and others use frame ids.
  - Small differences in seeking the starting position in a video.
  - Loaders not working properly in some scenarios.


## Implemented loaders
- [Decord](https://github.com/dmlc/decord)
  - CPU version, in the loader called `DecordVideo`.
  - GPU version, in the loader called `DecordVideoGPU`.
- Pillow
  - Standard [Pillow library](https://github.com/python-pillow/Pillow), in the loader called `Pillow`.
  - Optimized [Pillow-SIMD library](https://github.com/uploadcare/pillow-simd), in the loader called `PillowSIMD`.
- [OpenCV](https://opencv.org/)
  - Using the image loader, in `OpenCVImage`.
  - Using the VideoCapture video loader, in `OpenCVVideo`.
- [MMCV](https://mmcv.readthedocs.io/en/latest/).
  - Using the image loader, in `MMCVImage`.
  - Using the VideoReader video loader, in `MMCVVideo`.
- [PIMS](http://soft-matter.github.io/pims/dev/video.html).
  - With PyAV backend, in `PIMSPyAV`.
  - With ImageIO backend, in `PIMSImageIO`.
  - With MoviePy backend, in `PIMSMoviePy`.
- [FFmpeg](https://ffmpeg.org/), in the `FFmpeg` loader. Some other libraries use ffmpeg under the hood.
- [MoviePy](https://zulko.github.io/moviepy/), in the `MoviePy` loader.
- [PyTorchVideo](https://github.com/facebookresearch/pytorchvideo)
  - Frame-level decoder, in `PyTorchVideoFrames`.
  - PyAV decoder, in `PyTorchVideoPyAV`.
  - Torchvision decoder, in `PyTorchVideoTorchvision`.
  - Decord decoder, in `PyTorchVideoDecord`.
- [TorchVision VideoReader](https://pytorch.org/vision/main/generated/torchvision.io.VideoReader.html)
  - Video backend, in `TorchVisionVideoReader`.
  - PyAV backend, in `TorchVisionVideoReaderPyAV`.
  - CUDA backend, in `TorchVisionVideoReaderCUDA`.
- [TorchVision read_video](https://pytorch.org/vision/main/generated/torchvision.io.read_video.html), in the `TorchVisionReadVideo` video loader.
- [TorchVideo](https://github.com/torchvideo/torchvideo)
  - Using [GULP](https://github.com/achaiah/GulpIO), in the `TorchVideoGULP` loader.
  - Using PIL, in the `TorchVideoPIL` loader.
  - Using TorchVideo internal readers in `TorchVideoVideo`.
- [DALI](https://github.com/NVIDIA/DALI), in the `DALI` loader.

See `loaders.py` for details.


## Contributions to the project

Pull requests (corrections of bugs, more efficient loading, new features, or better documentation) are welcome! :smile:

Some possible additions are:
- [ ] Implement new loaders. Follow the loaders that are already implemented, and specifically follow the instructions at the top of the `loaders.py` file.
- [ ] Monitor I/O required
- [ ] Monitor memory (RAM or GPU memory) requirements
- [ ] Multiprocessing option, simulating multiple-worker PyTorch loader
- [ ] Improve reporting of results


