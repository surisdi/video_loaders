"""
Standardization instructions for all loaders:
- All loaders return (after the transform function) a torch tensor of shape (C, T, H, W). 
- The order of augmentations is: resize, center crop, normalize. Any of them can be disabled, independently of the 
others.
- The range of values *before* the normalization is [0, 1]. The mean and std values assume this range. If the 
normalization is disabled, the final range is therefore [0, 1].
- The random frames and random segments are the same for all loaders, and given by self.list_random_frames and 
self.list_random_segment_starts
"""
import abc
import itertools
import os
from abc import ABC, abstractmethod
from typing import Union, List, Iterator
import warnings

import PIL
import cv2
import ffmpeg
import numpy as np
import torch
import torchaudio
import torchvision
from PIL import Image

import transforms
from parameters import path_to_ffprobe, path_to_ffmpeg, parameters

loaders = parameters['loader_name']
if 'torchvideo_gulp' in loaders or 'torchvideo_pil' in loaders or 'torchvideo_video' in loaders:
    import torchvideo
if 'torchvideo_gulp' in loaders:
    import gulpio
    from gulpio import transforms as gulp_transforms
if 'moviepy' in loaders:
    import moviepy
if 'pytorchvideo_frames' in loaders or 'pytorchvideo_pyav' in loaders or 'pytorchvideo_torchvision' in loaders or \
        'pytorchvideo_decord' in loaders:
    from pytorchvideo.data.frame_video import FrameVideo
    from pytorchvideo.data.encoded_video import EncodedVideo
    import pytorchvideo.transforms
    import pytorchvideo.transforms.functional
if 'pims_pyav' in loaders or 'pims_imageio' in loaders or 'pims_moviepy' in loaders:
    import pims
if 'mmcv_video' in loaders or 'mmcv_image' in loaders:
    import mmcv
if 'decord_video' in loaders or 'decord_video_gpu' in loaders:
    import decord
if 'dali' in loaders:
    from nvidia.dali import pipeline_def
    import nvidia.dali.fn as fn
    import nvidia.dali.types as types

import utils
from utils import NotPossibleException


mean_norm = [0.485, 0.456, 0.406]
std_norm = [0.229, 0.224, 0.225]


def get_loader(loader_name, **kwargs):
    return {
        'decord_video': DecordVideo,
        'decord_video_gpu': DecordVideoGPU,
        'pillow': Pillow,
        'pillow_simd': PillowSIMD,
        'opencv_image': OpenCVImage,
        'opencv_video': OpenCVVideo,
        'mmcv_video': MMCVVideo,
        'mmcv_image': MMCVImage,
        'pims_pyav': PIMSPyAV,
        'pims_imageio': PIMSImageIO,
        'pims_moviepy': PIMSMoviePy,
        'ffmpeg': FFmpeg,
        'moviepy': MoviePy,
        'pytorchvideo_frames': PyTorchVideoFrames,
        'pytorchvideo_pyav': PyTorchVideoPyAV,
        'pytorchvideo_torchvision': PyTorchVideoTorchvision,
        'pytorchvideo_decord': PyTorchVideoDecord,
        'torchvision_videoloader': TorchVisionVideoReader,
        'torchvision_videoloader_pyav': TorchVisionVideoReaderPyAV,
        'torchvision_videoloader_cuda': TorchVisionVideoReaderCUDA,
        'torchvision_readvideo': TorchVisionReadVideo,
        'torchvideo_gulp': TorchVideoGULP,
        'torchvideo_pil': TorchVideoPIL,
        'torchvideo_video': TorchVideoVideo,
        'dali': DALI,
    }[loader_name](**kwargs)


class MyLoader(ABC):
    can_load_audio = True

    def __init__(self,
                 data_path,
                 resize=False,
                 short_side_size=224,
                 keep_aspect_ratio=True,
                 center_crop=False,
                 crop_size=224,
                 normalize=True,
                 clip_len=8,
                 frame_sample_rate=2,
                 load_format='random_frames',
                 list_random_frames=None,
                 list_random_segment_starts=None,
                 random_segment_duration=12,
                 random_segment_before_fps=False,
                 load_audio=False,
                 num_random_frames=None,  # unused
                 num_random_segments=None,  # unused
                 ):

        self.data_path = data_path
        self.resize = resize
        self.short_side_size = short_side_size
        self.keep_aspect_ratio = keep_aspect_ratio
        self.center_crop = center_crop
        self.crop_size = crop_size
        self.normalize = normalize
        self.clip_len = clip_len
        self.frame_sample_rate = frame_sample_rate
        self.load_format = load_format
        self.list_random_frames = list_random_frames
        self.list_random_segment_starts = list_random_segment_starts
        self.random_segment_duration = random_segment_duration
        self.random_segment_before_fps = random_segment_before_fps
        self.load_audio = load_audio

        list_transforms = []
        if self.resize:
            if keep_aspect_ratio:
                size = self.short_side_size
            else:
                size = (self.short_side_size, self.short_side_size)
            list_transforms.append(transforms.Resize(size, interpolation='bilinear'))
        if self.center_crop:  # Always applied after resizing, if resizing is enabled
            list_transforms.append(transforms.CenterCrop(self.crop_size))
        list_transforms.append(transforms.ClipToTensor())
        self.normalization_transform = transforms.Normalize(mean=mean_norm, std=std_norm)
        if self.normalize:
            list_transforms.append(self.normalization_transform)
        self.data_transform = transforms.Compose(list_transforms)

        if not self.can_load_audio and self.load_audio:
            raise NotPossibleException("Cannot load audio with this loader")

    @abstractmethod
    def read_video(self, video_path):
        pass

    def transform(self, frames):
        """
        This function should be implemented such that it returns a video with a CTHW format
        """
        return self.data_transform(frames)

    @staticmethod
    def get_frames_path(video_path):
        video_path = video_path.split('/')
        assert video_path[-2] == 'videos'
        video_path[-2] = 'frames'
        video_path[-1] = video_path[-1].replace('.mp4', '')
        video_path = '/'.join(video_path)
        return video_path

    def get_frame_indices(self, len_video, video_path=None, fps=None):
        if self.load_format == 'random_frames':
            frame_indices = [idx % len_video for idx in self.list_random_frames]
        elif self.load_format == 'random_segments':
            initial_fps = utils.get_fps(video_path) if fps is None else fps
            ratio = self.frame_sample_rate / initial_fps if self.frame_sample_rate != -1 else 1
            segment_starts = self.get_segment_starts(len_video, ratio)
            frame_indices = [round(start + idx / ratio) for start in segment_starts for idx in
                             range(self.random_segment_duration)]
        else:  # 'all_video'
            if self.frame_sample_rate == -1:
                frame_indices = list(range(len_video))
            else:
                initial_fps = utils.get_fps(video_path) if fps is None else fps
                ratio = self.frame_sample_rate / initial_fps
                frame_indices = list(np.round(np.linspace(0, len_video, num=round(len_video * ratio), endpoint=False)).
                                     astype(int))
        return frame_indices

    def get_segment_starts(self, len_video, ratio, return_in_original_fps=True):
        """
        Returns the starting frames of segments, where the frames are given in the original video's fps.
        If return_in_original_fps is True, the frames will be indexed wrt the original fps.
        """
        if self.random_segment_before_fps:
            max_seg_start = int(np.floor(len_video - self.random_segment_duration / ratio))
            segment_starts = [idx % max_seg_start for idx in self.list_random_segment_starts]
        else:
            max_seg_start = round(len_video * ratio) - self.random_segment_duration
            if return_in_original_fps:
                segment_starts = [round((idx % max_seg_start) / ratio) for idx in self.list_random_segment_starts]
            else:
                segment_starts = [idx % max_seg_start for idx in self.list_random_segment_starts]
        return segment_starts

    def add_current_frame(self, ratio, i):
        """
        When looping over all frames in a video, this function returns the number of times that frame should be added to
        the final video. If the final sampling rate is larger than the original fps, the frame should be added one or
        more times. If the final sampling rate is smaller than the original fps, the frame should be added zero or one
        times.
        """
        if self.frame_sample_rate == -1 or ratio == 1:
            to_add = 1
        elif ratio <= 1:
            closest_frame = np.round(i * ratio)
            condition = np.abs(i * ratio - closest_frame) <= np.abs((i - 1) * ratio - closest_frame) and \
                        np.abs(i * ratio - closest_frame) < np.abs((i + 1) * ratio - closest_frame)
            to_add = 1 if condition else 0
        else:  # The final sampling rate is > than the original, therefore some frames are repeated
            to_add = np.sum(np.round(np.array(range(int(np.ceil((i - 1) * ratio)),
                                                    int(np.ceil((i + 1) * ratio)))) / ratio) == i)
        return to_add

    @staticmethod
    def read_audio(video_path, start_time, end_time):
        """
        This function loads audio from a .wav file. It is used when the audio is not loaded by default by the loader.
        The purpose of this repository is not to compare audio loaders, so this default audio loader may not be the best
        one. If you want to use a different audio loader, you can override this function in your loader class.
        """
        audio_path = video_path.replace('videos', 'audios').replace('.mp4', '.wav')
        metadata = torchaudio.info(audio_path)
        sample_rate = metadata.sample_rate

        frame_offset = round(sample_rate * start_time)
        if end_time is None:
            num_frames = -1
        else:
            num_frames = round(sample_rate * (end_time - start_time))

        waveform, sample_rate = torchaudio.load(audio_path, frame_offset=frame_offset, num_frames=num_frames)

        return waveform

    def get_audio_segment(self, video_path, start, ratio, initial_fps=None):
        """
        Convenient function to load audio when we only have the start in frames (not in seconds). It deals with the
        conversion between fps and frames, and finds the corresponding audio segment.
        """
        initial_fps = utils.get_fps(video_path) if initial_fps is None else initial_fps
        if self.random_segment_before_fps:
            start_time = start / initial_fps
        else:
            start_time = start / initial_fps * ratio
        duration_time = self.random_segment_duration / initial_fps / ratio
        try:
            audio_segment = self.read_audio(video_path, start_time, start_time + duration_time)
        except RuntimeError:
            # Sometimes the video file is problematic, and the loaders return an incorrect number of frames
            start_time = start_time / 2
            audio_segment = self.read_audio(video_path, start_time, start_time + duration_time)
        return audio_segment


class DecordVideo(MyLoader):
    """https://github.com/dmlc/decord"""
    device = 'cpu'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Only resize has been done as part of read_video
        self.data_transform = transforms.Compose([t for t in self.data_transform.transforms
                                                  if not isinstance(t, transforms.Resize)])

    def read_video(self, video_path):
        """Load video content using Decord"""

        assert os.path.getsize(video_path) > 1 * 1024, 'Hanging issue'

        reader = decord.VideoReader if not self.load_audio else decord.AVReader

        ctx = {'cpu': decord.cpu(0), 'gpu': decord.gpu(0)}[self.device]
        if not self.resize:
            vr = reader(video_path, num_threads=1, ctx=ctx)
        else:
            if self.keep_aspect_ratio:
                # Find the width and height of the video
                new_width, new_height = utils.get_proportional_sizes(self.short_side_size, video_path)
            else: 
                new_width, new_height = self.short_side_size, self.short_side_size
            vr = reader(video_path, width=new_width, height=new_height, num_threads=1, ctx=ctx)

        decord.bridge.set_bridge('torch')

        len_video = len(vr)
        fps = (vr._AVReader__video_reader if self.load_audio else vr).get_avg_fps()
        frame_indices = self.get_frame_indices(len_video, video_path, fps)

        buffer = vr.get_batch(frame_indices)
        # If self.load_audio, the audio parameter is a list of audios associated to each retrieved frame
        audio, video = buffer if self.load_audio else (None, buffer)
        if self.load_audio:
            audio = torch.cat(audio, dim=1)  # Concatenate the audios for all the frames

        return video, audio

    def transform(self, frames):
        frames = self.data_transform(frames)
        return frames


class DecordVideoGPU(DecordVideo):
    """
    Note that only CPU versions are provided with PYPI now. Please build from source to enable GPU accelerator.
    https://github.com/dmlc/decord
    """
    device = 'gpu'


class Pillow(MyLoader):
    """
    Load from pre-extracted frames.
    The default transforms already consider the case where the frames are PIL images, using PIL transforms in that case.
    """
    can_load_audio = False

    def __init__(self, *args, **kwargs):
        assert "post" not in PIL.__version__, \
            "Pillow-SIMD is installed instead of Pillow. Specify the loader pillow_simd instead of pillow. " \
            "If you want to use Pillow, uninstall Pillow-SIMD and install Pillow."
        super().__init__(*args, **kwargs)

    def read_video(self, video_path):
        frames_path = self.get_frames_path(video_path)
        len_video = len(os.listdir(frames_path))  # Alternatively, read metadata for the len
        frame_indices = self.get_frame_indices(len_video, video_path)
        video = [Image.open(os.path.join(frames_path, '%07d.png' % idx)).convert('RGB') for idx in frame_indices]
        return video, None


class PillowSIMD(Pillow):
    """
    Load from pre-extracted frames.
    Fork from PIL: https://github.com/uploadcare/pillow-simd
    Pillow and Pillow-SIMD are not compatible. You can only have one of the installed. We could have implemented the two
    of them using the same class, and the one installed would be used. However, we decided to keep them separate
    explicitly so that the user knows which one is being used.
    """

    def __init__(self, *args, **kwargs):
        assert "post" in PIL.__version__, \
            "Pillow is installed instead of Pillow-SIMD. Specify the loader `pillow' instead of `pillow_simd'. " \
            "If you want to use Pillow-SIMD, uninstall Pillow and install Pillow-SIMD."
        super().__init__(*args, **kwargs)


class OpenCVImage(MyLoader):
    """Load from pre-extracted frames."""
    can_load_audio = False

    def read_video(self, video_path):
        frames_path = self.get_frames_path(video_path)
        len_video = len(os.listdir(frames_path))  # Alternatively, read metadata for the len
        frame_indices = self.get_frame_indices(len_video, video_path)
        # The ::-1 is to convert from BGR to RGB
        video = [cv2.imread(os.path.join(frames_path, '%07d.png' % idx))[:, :, ::-1] for idx in frame_indices]
        return video, None


class MMCVVideo(MyLoader):
    """MMCV does not have audio support."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def read_video(self, video_path):
        video_reader = mmcv.VideoReader(video_path)
        len_video = video_reader.frame_cnt
        if self.load_format == 'random_frames':
            frame_indices = self.get_frame_indices(len_video)
            video = [video_reader[idx] for idx in frame_indices]
            audio = None
            if self.load_audio:
                raise NotPossibleException('The default audio reader does not support random frames')
        elif self.load_format == 'random_segments':
            initial_fps = video_reader.fps
            ratio = self.frame_sample_rate / initial_fps if self.frame_sample_rate != -1 else 1
            segment_starts = self.get_segment_starts(len_video, ratio)
            video = []
            audio = [] if self.load_audio else None
            for start in segment_starts:
                video_segment = video_reader[start:start + round(self.random_segment_duration / ratio)]
                if self.frame_sample_rate != -1:
                    indices_resample = np.round(np.linspace(0, round(self.random_segment_duration / ratio),
                                                            num=round(self.random_segment_duration),
                                                            endpoint=False)).astype(int)
                    video_segment = [video_segment[idx] for idx in indices_resample]
                video.extend(video_segment)
                if self.load_audio:  # Use default implementation, no audio support
                    audio_segment = self.get_audio_segment(video_path, start, ratio, initial_fps)
                    audio.append(audio_segment)
        else:  # load_format == 'all_video'
            video = video_reader[:]
            # For some reason, video_reader.frame_cnt sometimes returns more frames than the correct number
            video = [v for v in video if v is not None]
            audio = None
            if self.load_audio:  # Use default implementation, no audio support
                audio = self.read_audio(video_path, 0, None)

        # Convert from BGR to RGB
        video = [v[..., ::-1] for v in video]

        """
        mmcv has some extra functionalities like the following:
        
        # obtain basic information
        print(len(video))
        print(video.width, video.height, video.resolution, video.fps)

        # iterate over all frames
        for frame in video:
            print(frame.shape)

        # read the next frame
        img = video.read()
        
        # cut a video clip
        mmcv.cut_video('test.mp4', 'clip1.mp4', start=3, end=10, vcodec='h264')
        
        # resize a video with the specified size
        mmcv.resize_video('test.mp4', 'resized1.mp4', (360, 240))
        
        The last two these do not return a video, just save a new video, so they are not useful in a data loading 
        context. In the background, they use ffmpeg.
        """

        return video, audio


class MMCVImage(MyLoader):
    can_load_audio = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def read_video(self, video_path):
        frames_path = self.get_frames_path(video_path)
        len_video = len(os.listdir(frames_path))  # Alternatively, read metadata for the len
        frame_indices = self.get_frame_indices(len_video, video_path)
        # The ::-1 is to convert from BGR to RGB
        video = [mmcv.imread(os.path.join(frames_path, '%07d.png' % idx))[..., ::-1] for idx in frame_indices]
        return video, None

    def transform(self, frames):
        transformed_frames = []
        for frame in frames:
            if self.resize:
                if self.keep_aspect_ratio:
                    new_h, new_w = transforms.get_resize_sizes(*frame.shape[:2], self.short_side_size)
                else:
                    new_h, new_w = self.short_side_size, self.short_side_size
                frame = mmcv.imresize(frame, (new_w, new_h))
            if self.center_crop:
                im_h, im_w = frame.shape[:2]
                h = w = self.crop_size
                if self.crop_size > im_w or self.crop_size > im_h:
                    error_msg = (
                        'Initial image size should be larger then '
                        'cropped size but got cropped sizes : ({w}, {h}) while '
                        'initial image is ({im_w}, {im_h})'.format(
                            im_w=im_w, im_h=im_h, w=w, h=h))
                    raise ValueError(error_msg)

                x1 = int(round((im_w - w) / 2.))
                y1 = int(round((im_h - h) / 2.))
                bboxes = np.array([x1, y1, x1 + self.crop_size, y1 + self.crop_size])
                frame = mmcv.imcrop(frame, bboxes)  # Note that this allows to crop multiple patches at once
            transformed_frames.append(frame)

        transformed_frames = np.stack(transformed_frames)  # (T, H, W, C)
        transformed_frames = transformed_frames.transpose([3, 0, 1, 2])  # (C, T, H, W)
        transformed_frames = transformed_frames / 255.
        transformed_frames = torch.tensor(transformed_frames)
        if self.normalize:
            transformed_frames = self.normalization_transform(transformed_frames)
        return transformed_frames


class OpenCVVideo(MyLoader):
    def read_video(self, video_path):
        # Open the video file
        cap = cv2.VideoCapture(video_path)

        # Get the frame rate of the video
        fps = cap.get(cv2.CAP_PROP_FPS)
        ratio = self.frame_sample_rate / fps if self.frame_sample_rate != -1 else 1

        if self.load_format == 'random_frames':
            len_video = utils.get_duration(video_path)
            frame_indices = self.get_frame_indices(len_video)
            # There are two ways of loading frames:
            """
            # 1) iterating over the video
            # Probably very inefficient unless the frame density is very high
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Start at frame 0
            video = []
            ret = True
            i = 0
            while ret:
                ret, frame = cap.read()
                if ret and i in frame_indices:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = Image.fromarray(frame)
                    video.append(frame)
                i += 1
            """
            # 2) using cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            video = []
            for frame_index in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                video.append(frame)
            audio = None
            if self.load_audio:
                raise NotPossibleException('OpenCVVideo does not support loading audio from random frames')

        elif self.load_format == 'random_segments':
            len_video = utils.get_duration(video_path)
            segment_starts = self.get_segment_starts(len_video, ratio)
            video = []
            audio = [] if self.load_audio else None
            for start in segment_starts:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                for i in range(start, round(start + self.random_segment_duration / ratio)):
                    video, ret = self.obtain_frame(cap, video, ratio, i)
                if self.load_audio:
                    audio_segment = self.get_audio_segment(video_path, start, ratio)
                    audio.append(audio_segment)

        else:  # load_format == 'all_video'
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Start at frame 0
            video = []
            ret = True
            i = 0
            while ret:
                video, ret = self.obtain_frame(cap, video, ratio, i)
                i += 1
            audio = None
            if self.load_audio:
                audio = self.read_audio(video_path, 0, None)

        # Release the video capture object
        cap.release()

        return video, audio

    def obtain_frame(self, cap, video, ratio, i):
        ret, frame = cap.read()
        to_add = self.add_current_frame(ratio, i)
        if ret and to_add > 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            video.extend([frame] * to_add)
        return video, ret


class PIMS(MyLoader, abc.ABC):
    """
    http://soft-matter.github.io/pims/dev/video.html
    """
    reader = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def read_video(self, video_path):
        reader = {'video': pims.Video,  # Same as pims.PyAVVideoReader or pims.PyAVReaderTimed
                  'imageio': pims.ImageIOReader,
                  'moviepy': pims.MoviePyReader}[self.reader]
        video_reader = reader(video_path)
        len_video = len(video_reader)
        frame_rate = video_reader.frame_rate
        ratio = self.frame_sample_rate / frame_rate if self.frame_sample_rate != -1 else 1

        if self.load_format == 'random_frames':
            frame_indices = self.get_frame_indices(len_video)
            video = [video_reader.get_frame(idx) for idx in frame_indices]
            audio = None
            if self.load_audio:
                raise NotPossibleException('PIMS does not support loading audio from random frames')

        elif self.load_format == 'random_segments':
            segment_starts = self.get_segment_starts(len_video, ratio)
            video = []
            audio = [] if self.load_audio else None
            for start in segment_starts:
                video_segment = []
                for i in range(start, start + round(self.random_segment_duration / ratio)):
                    video_segment, _ = self.obtain_frame(video_reader, video_segment, ratio, i)
                video.append(video_segment)
                if self.load_audio:
                    audio_segment = self.get_audio_segment(video_path, start, ratio)
                    audio.append(audio_segment)
            video = np.concatenate(video)

        else:  # load_format == 'all_video'
            video = []
            for i in range(len_video):
                video, end_video = self.obtain_frame(video_reader, video, ratio, i)
                if end_video:
                    break
            audio = None
            if self.load_audio:
                audio = self.read_audio(video_path, 0, None)

        return video, audio

    def obtain_frame(self, video_reader, video, ratio, i):
        # The different get_frame calls will not be independent, because there is some caching going on.
        # Sequential calls (or calls to frames close to the last frame) to get_frame will be faster than random calls.
        # If the first frame is very well into the video, the first call to get_frame will probably be slow
        to_add = self.add_current_frame(ratio, i)
        if to_add > 0:
            try:
                frame = video_reader.get_frame(i)
            except IndexError:
                """Sometimes, when the reader is imageio, depending on the formatting of the video, the duration of the
                video is not read properly (similarly to mmcv reader)"""
                return video, True
            video.extend([frame] * to_add)
        return video, False


class PIMSPyAV(PIMS):
    reader = 'video'


class PIMSImageIO(PIMS):
    """Slower than using AV. Implements interface with ffmpeg through a Pipe."""
    reader = 'imageio'


class PIMSMoviePy(PIMS):
    """Slower than using AV. Implements interface with ffmpeg through a Pipe."""
    reader = 'moviepy'


class FFmpeg(MyLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def read_video(self, video_path):
        len_video = utils.get_duration(video_path, method='ffprobe')
        frame_rate = utils.get_fps(video_path, method='ffprobe')
        ratio = self.frame_sample_rate / frame_rate if self.frame_sample_rate != -1 else 1

        if self.load_format == 'random_frames':
            frame_indices = self.get_frame_indices(len_video)
            fps, width, height = self.get_metadata(video_path)
            video = [self.read_frame_time(video_path, idx, fps, width, height) for idx in frame_indices]
            video = np.transpose(np.stack(video), (3, 0, 1, 2))
            audio = None
            if self.load_audio:
                raise NotPossibleException('FFmpeg does not support loading audio from random frames')

        elif self.load_format == 'random_segments':
            segment_starts = self.get_segment_starts(len_video, ratio, return_in_original_fps=False)
            video = []
            audio = [] if self.load_audio else None
            for start in segment_starts:
                duration_frames = self.random_segment_duration / (ratio if self.random_segment_before_fps else 1)
                video_seg, audio_seg = self.read_video_start_end(video_path, start, start + duration_frames,
                                                                 self.random_segment_before_fps)
                video.append(video_seg)
                if self.load_audio:
                    audio.append(audio_seg)
            video = torch.cat(video, dim=1)

        else:  # load_format == 'all_video'
            video, audio = self.read_video_start_end(video_path, 0, None)

        return video, audio

    def ffmpeg_transforms(self, v, width, height):
        if self.resize:
            if self.keep_aspect_ratio:
                v = ffmpeg.filter(v, filter_name='scale',
                                  w=f"if(gt(iw,ih),-1,{self.short_side_size})",
                                  h=f"if(gt(iw,ih),{self.short_side_size},-1)")
                rescale_factor = self.short_side_size / min(width, height)
                width = round(width * rescale_factor)
                height = round(height * rescale_factor)
            else:
                v = ffmpeg.filter(v, filter_name='scale', w=self.short_side_size, h=self.short_side_size)
                width = height = self.short_side_size

        if self.center_crop:
            aw, ah = 0.5, 0.5  # Center crop
            v = ffmpeg.crop(v,
                            '(iw - {})*{}'.format(self.crop_size, aw),
                            '(ih - {})*{}'.format(self.crop_size, ah),
                            str(self.crop_size),
                            str(self.crop_size))
            width = height = self.crop_size

        return v, width, height

    def read_video_start_end(self, video_path, start_frame, end_frame, frame_before_fps=True):
        # If frame_before_fps is True, the start and end frames are given in the original frame rate.

        # Get width and height to reshape later, as well as fps
        fps, width, height = self.get_metadata(video_path)

        if not frame_before_fps:
            fps = self.frame_sample_rate
        start_seconds = start_frame / fps if start_frame is not None else 0
        end_seconds = end_frame / fps if end_frame is not None else None

        if end_frame is None:
            v = ffmpeg.input(video_path, ss=start_seconds)
            if self.frame_sample_rate != -1:
                v = ffmpeg.filter(v, filter_name='fps', fps=self.frame_sample_rate)
        else:
            v = ffmpeg.input(video_path)

            """
            This commented out code is equivalent to trimming the video later, but in seconds instead of frame idx. 
            I implement it with trim because it is more flexible: it allows to select the time before and after changing 
            fps. The problem with trim is that then the audio cannot be extracted directly from v.
            
            v_initial = ffmpeg.input(video_path, ss=start_seconds, t=end_seconds - start_seconds)
            """

            if frame_before_fps:
                v = v.trim(start_frame=start_frame, end_frame=end_frame)
                if self.frame_sample_rate != -1:
                    v = ffmpeg.filter(v, filter_name='fps', fps=self.frame_sample_rate)
            else:
                if self.frame_sample_rate != -1:
                    v = ffmpeg.filter(v, filter_name='fps', fps=self.frame_sample_rate)
                v = v.trim(start_frame=start_frame, end_frame=end_frame)  # This accounts for the fps change already

        # For some reason this is necessary to do the trimming. 
        # See https://github.com/kkroening/ffmpeg-python/issues/184
        v = v.setpts('PTS-STARTPTS')

        v, width, height = self.ffmpeg_transforms(v, width, height)

        out, _ = (
            v.output('pipe:', format='rawvideo', pix_fmt='rgb24').
            run(capture_stdout=True, quiet=True, cmd=path_to_ffmpeg)
        )
        video = np.frombuffer(out, np.uint8)
        video = video.reshape([-1, height, width, 3])
        video = torch.from_numpy(np.array(video))
        video = video.permute(3, 0, 1, 2)  # [C, T, H, W]

        audio = None
        if self.load_audio:
            v = ffmpeg.input(video_path, ss=start_seconds) if end_seconds is None else \
                ffmpeg.input(video_path, ss=start_seconds, t=end_seconds - start_seconds)
            a = v.audio
            out_audio, _ = (
                a.output('pipe:', format='s16le', acodec='pcm_s16le', ac=1).  # ar=sample_rate).
                run(capture_stdout=True, quiet=True, cmd=path_to_ffmpeg)
            )
            audio = np.frombuffer(out_audio, np.int16)
            audio = torch.from_numpy(np.array(audio))

        return video, audio

    @staticmethod
    def get_metadata(video_path):
        probe = ffmpeg.probe(video_path, cmd=path_to_ffprobe)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        width = video_info['width']
        height = video_info['height']
        fps_num, fps_den = video_info['r_frame_rate'].split('/')
        fps = float(fps_num) / float(fps_den)
        return fps, width, height

    def read_frame_time(self, video_path, frame_idx, fps, width, height=None):
        frame_time = frame_idx / fps

        v = ffmpeg.input(video_path, ss=frame_time)

        if self.frame_sample_rate != -1:
            v = ffmpeg.filter(v, filter_name='fps', fps=self.frame_sample_rate)

        v, width, height = self.ffmpeg_transforms(v, width, height)

        out, _ = (
            v.output('pipe:', format='rawvideo', pix_fmt='rgb24', vframes=1).
            run(capture_stdout=True, quiet=True, cmd=path_to_ffmpeg)
        )
        frame = np.frombuffer(out, np.uint8)
        frame = frame.reshape([height, width, 3])
        return frame

    def transform(self, frames):
        frames = frames / 255.
        frames = torch.tensor(frames)
        if self.normalize:
            frames = self.normalization_transform(frames)
        return frames


class MoviePy(MyLoader):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.random_segment_before_fps:
            raise NotPossibleException('random_segment_before_fps=False does not apply here, because the times are '
                                       'given in seconds, not frame number. The order of set_fps and subclip(s,e) does '
                                       'not matter. First, the clip is moved to the start, then it is sampled at the '
                                       'given fps.')

    def read_video(self, video_path):
        # If we do not use the context manager, remember to clip.close()
        with moviepy.editor.VideoFileClip(video_path) as clip:
            len_video = utils.get_duration(video_path, method='ffprobe')
            frame_rate = clip.fps
            ratio = self.frame_sample_rate / frame_rate if self.frame_sample_rate != -1 else 1

            if self.frame_sample_rate == -1:
                video_clip = clip
            else:
                video_clip = clip.with_fps(self.frame_sample_rate)
            width = clip.w
            height = clip.h
            video_clip = self.moviepy_transforms(video_clip, width, height)

            if self.load_format == 'random_frames':
                frame_indices = self.get_frame_indices(len_video)
                video = [video_clip.get_frame(idx / frame_rate) for idx in frame_indices]
                video = np.stack(video)
                audio = None
                if self.load_audio:
                    raise NotPossibleException('Moviepy does not support loading audio from random frames')

            elif self.load_format == 'random_segments':
                segment_starts = self.get_segment_starts(len_video, ratio)
                video = []
                audio = [] if self.load_audio else None
                for start in segment_starts:
                    start_time = start / frame_rate
                    end_time = (start + self.random_segment_duration / ratio) / frame_rate
                    video_seg = video_clip.subclip(start_time, end_time)
                    video_seg_array = np.array(list(video_seg.iter_frames()))

                    video.append(video_seg_array)
                    if self.load_audio:
                        audio.append(video_seg.audio.to_soundarray())
                video = np.concatenate(video)

            else:  # load_format == 'all_video'
                video = np.array(list(video_clip.iter_frames()))
                audio = None
                if self.load_audio:
                    audio = video_clip.audio.to_soundarray()

            video_clip.close()

            return video, audio

    def moviepy_transforms(self, clip, width, height):
        if self.resize:
            if self.keep_aspect_ratio:
                new_width, new_height = utils.get_proportional_sizes(self.short_side_size, width=width, height=height)
            else:
                new_width, new_height = self.short_side_size, self.short_side_size
            clip = clip.resize((new_width, new_height))
            width, height = new_width, new_height
        if self.center_crop:
            clip = clip.crop(x_center=width // 2, y_center=height // 2, width=self.crop_size, height=self.crop_size)
        return clip

    def transform(self, frames):
        frames = frames / 255.
        frames = frames.transpose(3, 0, 1, 2)  # CTHW
        if self.normalize:
            frames = self.normalization_transform(torch.tensor(frames))
        frames = torch.tensor(frames)
        return frames


class PyTorchVideo(MyLoader, abc.ABC):
    """
    https://github.com/facebookresearch/pytorchvideo
    PytorchVideo has some other convenient classes, for example to organize the sampling of clips from a video in
    https://github.com/facebookresearch/pytorchvideo/blob/main/pytorchvideo/data/clip_sampling.py
    """
    decoder = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fps_transform = None

    def read_video(self, video_path):
        if self.decoder == 'frames':
            fps = utils.get_fps(video_path)
            video_path_frames = video_path.replace('videos', 'frames').replace('.mp4', '')
            # Optionally set multithread_io to False
            video_reader = FrameVideo.from_directory(video_path_frames, fps, multithreaded_io=True)
        else:
            video_reader = EncodedVideo.from_path(video_path, decode_audio=self.load_audio, decoder=self.decoder)
            fps = self.get_fps(video_reader, video_path)

        len_video = video_reader.duration * fps
        ratio = self.frame_sample_rate / fps if self.frame_sample_rate != -1 else 1

        if self.load_format == 'random_frames':
            frame_indices = self.get_frame_indices(len_video)
            video = [video_reader.get_clip(idx / fps, (idx + 1) / fps) for idx in frame_indices]
            video = torch.cat([v['video'] for v in video], axis=1)
            audio = None
            if self.load_audio:
                raise NotPossibleException('PyTorchVideo does not support loading audio from random frames')

        elif self.load_format == 'random_segments':
            segment_starts = self.get_segment_starts(len_video, ratio)
            video = []
            audio = [] if self.load_audio else None
            for start in segment_starts:
                video_dict = video_reader.get_clip(start / fps, (start + self.random_segment_duration / ratio) / fps)
                video_seg = video_dict['video']
                if self.frame_sample_rate != -1:
                    video_seg = pytorchvideo.transforms.functional.uniform_temporal_subsample\
                        (video_seg, self.random_segment_duration)
                video.append(video_seg)
                if self.load_audio:
                    audio_segment = video_dict['audio']
                    audio.append(audio_segment)

            video = torch.cat(video, dim=1)

        else:  # load_format == 'all_video'
            video_dict = video_reader.get_clip(0, video_reader.duration)
            video = video_dict['video']
            if self.frame_sample_rate != -1:
                """
                The uniform temporal subsampling is also very convenient when we have different lengths of videos and we
                want to sample the same number of frames from each video. Not this exact setting.
                """
                num_frames_final = int(video_reader.duration * self.frame_sample_rate)
                video = pytorchvideo.transforms.functional.uniform_temporal_subsample(video, num_frames_final)
            audio = video_dict['audio'] if self.load_audio else None

        video_reader.close()

        return video, audio

    def transform(self, frames):
        frames = frames / 255.
        if self.resize:
            if self.keep_aspect_ratio:
                frames = pytorchvideo.transforms.ShortSideScale(self.short_side_size)(frames)
            else:
                frames = frames.permute(1, 2, 3, 0).numpy()  # T, H, W, C
                frames = transforms.resize_clip(frames, (self.short_side_size, self.short_side_size))
                frames = torch.tensor(np.stack(frames)).permute(3, 0, 1, 2)  # C, T, H, W
        if self.center_crop:
            # For some reason, this is applied on top of the dictionary, not the tensor
            # 1 means center crop
            frames = pytorchvideo.transforms.functional.uniform_crop(frames, self.crop_size, 1)
        if self.normalize:
            frames = pytorchvideo.transforms.Normalize(mean=mean_norm, std=std_norm)(frames)
        return frames
    
    @staticmethod
    def get_fps(video_reader, video_path):
        raise NotImplementedError


class PyTorchVideoFrames(PyTorchVideo):
    can_load_audio = False
    decoder = 'frames'

    @staticmethod
    def get_fps(video_reader, video_path):
        return video_reader._fps


class PyTorchVideoPyAV(PyTorchVideo):
    decoder = 'pyav'

    @staticmethod
    def get_fps(video_reader, video_path):
        # Ideally there should be an attribute average_rate, which is what PyAV offers. But not implemented in
        # pytorchvideo
        return utils.get_fps(video_path)


class PyTorchVideoTorchvision(PyTorchVideo):
    decoder = 'torchvision'

    @staticmethod
    def get_fps(video_reader, video_path):
        return video_reader._fps


class PyTorchVideoDecord(PyTorchVideo):
    decoder = 'decord'

    @staticmethod
    def get_fps(video_reader, video_path):
        return video_reader._fps


class TorchVisionVideoReader(MyLoader):
    """
    https://github.com/pytorch/vision/blob/main/torchvision/io/video_reader.py
    https://pytorch.org/vision/main/generated/torchvision.io.VideoReader.html
    https://pytorch.org/vision/main/auto_examples/plot_video_api.html
    """
    backend = 'video_reader'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.frame_sample_rate != -1 and self.load_format != 'random_frames':
            raise NotPossibleException(
                'TorchVisionVideoReader and TorchVisionReadVideo do not support changing the frame sample rate. A '
                'post-processing operation could be used to do this, but it is not implemented '
                'here because the torchvision Video API does not have an implementation for it.')

        # Implement same transforms, but using torchvision library
        list_transforms = []
        if self.resize:
            if self.keep_aspect_ratio:
                size = self.short_side_size
            else:
                size = (self.short_side_size, self.short_side_size)
            bilinear = torchvision.transforms.InterpolationMode.BILINEAR
            list_transforms.append(torchvision.transforms.Resize(size, interpolation=bilinear))
        if self.center_crop:  # Always applied after resizing, if resizing is enabled
            list_transforms.append(torchvision.transforms.CenterCrop(self.crop_size))
        if self.normalize:
            list_transforms.append(torchvision.transforms.Normalize(mean=mean_norm, std=std_norm))
        self.data_transform = transforms.Compose(list_transforms)

    def read_video(self, video_path):
        torchvision.set_video_backend(self.backend)

        reader = torchvision.io.VideoReader(video_path, "video")
        reader_audio = torchvision.io.VideoReader(video_path, "audio") if self.load_audio else None
        metadata = reader.get_metadata()
        fps = metadata['video']['fps'][0]
        duration = metadata['video']['duration'][0]
        len_video = int(duration * fps)
        ratio = self.frame_sample_rate / fps if self.frame_sample_rate != -1 else 1

        if self.load_format == 'random_frames':
            frame_indices = self.get_frame_indices(len_video)
            video = []
            audio = [] if self.load_audio else None
            for idx in frame_indices:
                second_idx = idx / fps
                reader.seek(second_idx)
                frame = next(reader)
                video.append(frame['data'])
                if self.load_audio:
                    reader_audio.seek(second_idx)
                    audio_segment = next(reader_audio)
                    audio.append(audio_segment['data'])

        elif self.load_format == 'random_segments':
            # If backend is pyav, this may not work well. You will get a warning that says that "Accurate seek is not 
            # implemented for pyav backend"
            segment_starts = self.get_segment_starts(len_video, ratio)
            video = []
            audio = [] if self.load_audio else None
            for start in segment_starts:
                start_seconds = start / fps
                end_seconds = (start + self.random_segment_duration / ratio) / fps
                # We use this if we want to read based on duration or end/start times
                v = reader.seek(start_seconds)
                for frame in itertools.takewhile(lambda x: x['pts'] <= end_seconds, v):
                    if frame['pts'] < start_seconds:
                        # This is a quick fix for pyav, which does not support accurate seek
                        continue
                    video.append(frame['data'])
                if self.load_audio:
                    a = reader_audio.seek(start_seconds)
                    for frame in itertools.takewhile(lambda x: x['pts'] <= end_seconds, a):
                        audio.append(frame['data'])
                """
                # We can use this if we want to read a specific number of frames. 
                for frame in itertools.islice(reader.seek(start_seconds), num_frames_load):
                    video.append(frame['data'])
                """

        else:  # load_format == 'all_video'
            video = []
            audio = [] if self.load_audio else None
            for frame in reader:
                video.append(frame['data'])
            if self.load_audio:
                for frame in reader_audio:
                    audio.append(frame['data'])

        video = torch.stack(video)
        return video, audio

    def transform(self, frames):
        frames = frames / 255
        frames = self.data_transform(frames)
        frames = frames.permute(1, 0, 2, 3)  # C, T, H, W
        return frames


class TorchVisionVideoReaderPyAV(TorchVisionVideoReader):
    """
    Note that, according to a Warning in the documentation, "Accurate seek is not implemented for pyav backend".
    May be only useful for reading the whole video.
    """
    backend = 'pyav'


class TorchVisionVideoReaderCUDA(TorchVisionVideoReader):
    backend = 'cuda'


class TorchVisionReadVideo(TorchVisionVideoReader):
    """
    https://github.com/pytorch/vision/blob/main/torchvision/io/video.py
    https://pytorch.org/vision/stable/generated/torchvision.io.read_video.html#torchvision.io.read_video
    Same library as TorchVisionVideoReader, but different implementation. See https://pytorch.org/vision/stable/io.html

    They use the FFmpeg C API under the hood for decoding.

    We inherit from TorchVisionVideoReader because transforms are torchvision transforms, and fps limitations are the
    same.
    """

    def read_video(self, video_path):
        len_video = utils.get_duration(video_path)  # Using some other implementation to get duration
        fps = utils.get_fps(video_path)  # Using some other implementation to get fps
        ratio = self.frame_sample_rate / fps if self.frame_sample_rate != -1 else 1

        if self.load_format == 'random_frames':
            frame_indices = self.get_frame_indices(len_video)
            video = []
            audio = [] if self.load_audio else None
            for idx in frame_indices:
                vid, aud, metadata = torchvision.io.read_video(video_path, start_pts=idx / fps, end_pts=idx / fps,
                                                               pts_unit='sec', output_format='TCHW')
                video.append(vid)
                if self.load_audio:
                    audio.append(aud)
            video = torch.cat(video)

        elif self.load_format == 'random_segments':
            segment_starts = self.get_segment_starts(len_video, ratio)
            video = []
            audio = [] if self.load_audio else None
            for start in segment_starts:
                start_seconds = start / fps
                end_seconds = (start + self.random_segment_duration / ratio) / fps
                # This function returns frames a bit outside of the [start_seconds, end_seconds] interval (from before 
                # of the interval)
                video_segment, audio_segment, metadata = torchvision.io.read_video(video_path, start_pts=start_seconds,
                                                                                   end_pts=end_seconds, pts_unit='sec',
                                                                                   output_format='TCHW')
                video.append(video_segment)
                if self.load_audio:
                    audio.append(audio_segment)

            video = torch.cat(video)

        else:  # load_format == 'all_video'
            video, audio, metadata = torchvision.io.read_video(video_path, output_format='TCHW')
            audio = audio if self.load_audio else None

        return video, audio


class TorchVideo(MyLoader, abc.ABC):
    """
    https://github.com/torchvideo/torchvideo
    Good for pre-extracted frames
    No audio support

    Under the hood, the samplers in torchvideo work by providing slices or lists as indices
    (https://torchvideo.readthedocs.io/en/latest/samplers.html?highlight=samplers#samplers)
    To make this code as similar as possible to their samplers, but adapted to our setting, we use the same approach.
    """

    approach = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "post" not in PIL.__version__:
            warnings.warn(
                "Pillow is installed instead of Pillow-SIMD. TorchVideo is supposed to work with Pillow-SIMD. It can "
                "be unfair to benchmark without Pillow-SIMD."
            )

        list_transforms = []
        if self.resize:
            if self.keep_aspect_ratio:
                size = self.short_side_size
            else:
                size = (self.short_side_size, self.short_side_size)
            list_transforms.append(torchvideo.transforms.ResizeVideo(size))  # default interpolation is bilinear
        if self.center_crop:
            list_transforms.append(torchvideo.transforms.CenterCropVideo(self.crop_size))
        list_transforms.append(torchvideo.transforms.CollectFrames())
        # list_transforms.append(torchvideo.transforms.PILVideoToTensor(rescale=True))
        list_transforms.append(utils.PILVideoToTensor(rescale=True))
        if self.normalize:
            list_transforms.append(torchvideo.transforms.NormalizeVideo(mean=mean_norm, std=std_norm))
        self.data_transform = torchvision.transforms.Compose(list_transforms)

    def read_video(self, video_path):
        # ffprobe is how torchvideo implements this too, in torchvideo.internal.readers._get_videofile_frame_count
        len_video = utils.get_duration(video_path, method='ffprobe')
        fps = utils.get_fps(video_path)  # Using some other implementation to get fps
        ratio = self.frame_sample_rate / fps if self.frame_sample_rate != -1 else 1
        frame_step = int(round(1/ratio))

        if self.load_format == 'random_frames':
            frame_indices = self.get_frame_indices(len_video)
            # load_frames does not take a list of frames, it requires a slice, so we have to retrieve every frame 
            # separately
            video = [self.load_frames(slice(f_idx, f_idx + 1, 1), video_path) for f_idx in frame_indices]
            video = [frame for segment in video for frame in segment]  # There's only one frame in the segment
            audio = None
            if self.load_audio:
                raise NotPossibleException('The default audio reader does not support random frames')

        # From the different readers, only GULP requires a slice. In the other approaches it does not matter. Actually, 
        # they convert to list from slice, so a list is fine. We code this using slices, which work for all readers.
        elif self.load_format == 'random_segments':
            # Following https://torchvideo.readthedocs.io/en/latest/_modules/torchvideo/samplers.html#ClipSampler
            duration_segment_input_frames = int(round(self.random_segment_duration/ratio))
            segment_starts = self.get_segment_starts(len_video, ratio, return_in_original_fps=False)
            video = []
            audio = [] if self.load_audio else None
            for start in segment_starts:
                frame_indices = slice(start, start + duration_segment_input_frames, frame_step)   # This is what ClipSampler returns
                video_segment = self.load_frames(frame_indices, video_path)
                video.append(video_segment)

                if self.load_audio:  # Use default implementation, no audio support
                    audio_segment = self.get_audio_segment(video_path, start, ratio, fps)
                    audio.append(audio_segment)

            video = [frame for segment in video for frame in segment]

        else:  # load_format == 'all_video'
            # This is just copied from the torchvideo FullVideoSampler in 
            # https://torchvideo.readthedocs.io/en/latest/_modules/torchvideo/samplers.html#FullVideoSampler
            frame_indices = slice(0, len_video, frame_step)
            audio = None
            if self.load_audio:  # Use default implementation, no audio support
                audio = self.read_audio(video_path, 0, None)

            video = self.load_frames(frame_indices, video_path)

        return video, audio

    def load_frames(self, frame_idx, video_path, *args):
        raise NotImplementedError

    def transform(self, frames):
        frames = self.data_transform(frames)  # Output has shape CTHW
        frames = torch.tensor(np.stack(frames))
        return frames


class TorchVideoGULP(TorchVideo):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        path_gulp_dir = os.path.join(self.data_path, 'gulp')
        self.gulp_dir = gulpio.GulpDirectory(path_gulp_dir)

        # Information can be obtained using:
        # self.items = self.gulp_dir.merged_meta_dict.items()

        image_transforms = []
        if self.resize:
            if self.keep_aspect_ratio:
                size = self.short_side_size
            else:
                size = (self.short_side_size, self.short_side_size)
            image_transforms.append(gulp_transforms.Scale(size))
        if self.center_crop:
            image_transforms.append(gulp_transforms.CenterCrop(self.crop_size))  # Uses OpenCV
        if self.normalize:
            image_transforms.append(gulp_transforms.Normalize(mean=mean_norm, std=std_norm))
        video_transforms = []
        self.data_transform = gulp_transforms.ComposeVideo(image_transforms, video_transforms)

    def load_frames(self, frame_idx: slice, video_path=None, id_: str=None) -> np.ndarray:
        """
        From torchvideo.datasets.gulp_folder_dataset.py
        Requires pre-extracting frames in a binary format of concatenated JPEGs (see https://github.com/achaiah/GulpIO)
        """
        id_ = id_ or video_path.split('/')[-1].split('.')[0]
        frames, _ = self.gulp_dir[id_, frame_idx]
        return frames
    
    def transform(self, frames):
        frames = [frame / 255. for frame in frames]
        frames = self.data_transform(frames)  # Output has shape CTHW
        frames = np.stack(frames)
        frames = np.transpose(frames, (3, 0, 1, 2))
        frames = torch.tensor(frames)
        return frames
    

class TorchVideoPIL(TorchVideo):
    def load_frames(self, frames_idx: Union[slice, List[slice], List[int]], video_path, *args):
        """
        From torchvideo.datasets.image_folder_video_dataset.py
        Requires pre-computed frames
        """
        frame_numbers = torchvideo.samplers.frame_idx_to_list(frames_idx)
        frames_path = self.get_frames_path(video_path)
        filepaths = [os.path.join(frames_path, '%07d.png' % idx) for idx in frame_numbers]
        frames = (PIL.Image.open(str(path)).convert('RGB') for path in filepaths if os.path.isfile(path))
        # shape: (n_frames, height, width, channels)
        return frames


class TorchVideoVideo(TorchVideo):
    @staticmethod
    def load_frames(frame_idx: Union[slice, List[slice], List[int]], video_path, *args):
        """
        From torchvideo.datasets.video_folder_dataset.py
        Each video is a single example in the dataset."""
        from torchvideo.internal.readers import default_loader

        frames = default_loader(video_path, frame_idx)
        return frames


class DALI(MyLoader):
    """
    https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/video/video_reader_simple_example.html
    https://github.com/NVIDIA/DALI
    DALI is the spiritual successor of NVVL.

    Note that the intended use of this loader is a bit different than what we implement here. A pipe should be created
    just once, and then read from it. However, for this project, we want to load specific videos, and in order to 
    control what videos are loaded, we need to create separate pipes for each one. The pipe creation adds some tenths of 
    a second more for every read video.

    We use num_threads=1 because in this specific implementation we did not see any speedup with more threads. Not even
    when we modified the batch size or the sequence length. A different implementation (for example, using a single pipe
    for all videos) could benefit from more threads.
    """

    num_threads=1

    def read_video(self, video_path):
            
        len_video = utils.get_duration(video_path) 
        fps = utils.get_fps(video_path)
        ratio = self.frame_sample_rate / fps if self.frame_sample_rate != -1 else 1

        if self.load_format in ['random_frames', 'random_segments']:

            if self.load_format == 'random_frames':
                duration_segment_input_frames = sequence_length = 1  # 'segments' of 1 frame
                segment_starts = self.get_frame_indices(len_video)
                stride = 1

            else:  # random_segments
                sequence_length = self.random_segment_duration
                duration_segment_input_frames = int(round(self.random_segment_duration/ratio))
                segment_starts = self.get_segment_starts(len_video, ratio, return_in_original_fps=False)
                stride = int(round(1/ratio))

            # Create file_list with a list of "file label start_frame end_frame" lines. Then save to /tmp
            path_tmp = '/tmp/file_list.txt'
            with open(path_tmp, 'w') as f:
                for start in segment_starts:
                    f.write(f'{video_path} 0 {start} {start + duration_segment_input_frames}\n')

            @pipeline_def
            def video_pipe():
                func = fn.readers.video_resize if self.resize else fn.readers.video
                params = {
                    'device': 'gpu',
                    'file_list': path_tmp,
                    'sequence_length': sequence_length,
                    'file_list_frame_num': True,
                    'random_shuffle': False,
                    'stride': stride,
                }
                if self.resize:
                    if self.keep_aspect_ratio:
                        params.update({
                            'resize_shorter': self.short_side_size,
                        })
                    else:
                        params.update({
                            'size': (self.short_side_size, self.short_side_size),
                        })
                video = func(**params)

                if self.center_crop:
                    raise NotPossibleException("The crop function in DALI does not work when loading from a file list")
                    # video = fn.crop(video, crop=(self.crop_size, self.crop_size), crop_pos_x=0.5, crop_pos_y=0.5)
                video = [v / types.Constant(np.float32(255)) for v in video]
                if self.normalize:
                    raise NotPossibleException("The normalize function in DALI does not work when loading from a file \
                                               list")
                    # video = [fn.normalize(v, mean=torch.tensor(mean_norm)[None, None, None], 
                return *video,
            
            pipe = video_pipe(batch_size=1, num_threads=self.num_threads, device_id=0)
            pipe.build()

            video = []
            for iter in range(len(segment_starts)):
                pipe_out = pipe.run()
                video_segment = pipe_out[0].as_cpu().as_array()
                assert video_segment.shape[0] == 1
                video_segment = video_segment[0]
                video.append(video_segment)
            video = np.concatenate(video, axis=0)

            os.remove(path_tmp)

            audio = self.get_audio_segments(video_path, ratio, fps) \
                if self.load_audio else None

        else:  # load_format == 'all_video'
            stride = int(round(1/ratio))
            sequence_length = int(len_video//stride)

            @pipeline_def
            def video_pipe():
                func = fn.readers.video_resize if self.resize else fn.readers.video
                params = {
                    'device': 'gpu',
                    'filenames': [video_path],
                    'sequence_length': sequence_length,
                    'random_shuffle': False,
                    'stride': stride,
                }
                if self.resize:
                    if self.keep_aspect_ratio:
                        params.update({
                            'resize_shorter': self.short_side_size,
                        })
                    else:
                        params.update({
                            'size': (self.short_side_size, self.short_side_size),
                        })
                video = func(**params)
                if self.center_crop:
                    video = fn.crop(video, crop=(self.crop_size, self.crop_size), crop_pos_x=0.5, crop_pos_y=0.5)
                video = video / types.Constant(np.float32(255))
                if self.normalize:
                    video = fn.normalize(video, mean=torch.tensor(mean_norm)[None, None, None], 
                                         stddev=torch.tensor(std_norm)[None, None, None], axis_names="FHW")
                return video

            pipe = video_pipe(batch_size=1, num_threads=self.num_threads, device_id=0)
            pipe.build()

            # A single iteration is enough, because the sequence length is the whole video
            # If the video is too large to load into memory, then we would have to reduce the sequence length, and
            # run pipe.run() several times.

            pipe_out = pipe.run()
            video = pipe_out[0].as_cpu().as_array()
            assert video.shape[0] == 1
            video = video[0]

            audio = None
            if self.load_audio:
                audio = self.read_audio(video_path)

        return video, audio
    
    def read_audio(self, video_path, return_sr=False):
        """
        https://docs.nvidia.com/deeplearning/dali/user-guide/docs/operations/nvidia.dali.fn.decoders.audio.html
        Does not support GPU decoding
        """
        audio_path = video_path.replace('videos', 'audios').replace('.mp4', '.wav')
        # Create file_list with a list of "file label" lines. Then save to /tmp
        path_tmp = '/tmp/file_list_audio.txt'
        with open(path_tmp, 'w') as f:
            f.write(f'{audio_path} 0\n')
        @pipeline_def
        def audio_decoder_pipe():
            encoded, _ = fn.readers.file(file_list=path_tmp)
            audio, sr = fn.decoders.audio(encoded, dtype=types.INT16)
            return audio, sr
        pipe = audio_decoder_pipe(batch_size=1, num_threads=self.num_threads, device_id=0)
        pipe.build()
        audio_cpu, sampling_rate = pipe.run()
        audio = audio_cpu.as_array()
        os.remove('/tmp/file_list_audio.txt')
        if return_sr:
            return audio, sampling_rate
        return audio

        
    def get_audio_segments(self, video_path, ratio, initial_fps=None):
        initial_fps = utils.get_fps(video_path) if initial_fps is None else initial_fps
        if self.random_segment_before_fps:
            start_time = [start / initial_fps for start in self.list_random_segment_starts]
        else:
            start_time = [start / initial_fps * ratio for start in self.list_random_segment_starts]
        duration_time = self.random_segment_duration / initial_fps / ratio

        # DALI audio reader does not have a start and end time, so we need to read the whole audio and then slice it
        audio, sr = self.read_audio(video_path, return_sr=True)
        sr = float(str(sr).split('[')[1].split(']')[0])

        audio_segments = []
        for start in start_time:
            start_audio_frame = int(start * sr)
            end_audio_frame = int((start + duration_time) * sr)
            audio_segment = audio[start_audio_frame:end_audio_frame]
            audio_segments.append(audio_segment)
        return audio_segments
    
    def transform(self, frames):
        """
        In this loader, the transform is done as part of the loading.
        """
        return torch.tensor(frames).permute(3, 0, 1, 2)
