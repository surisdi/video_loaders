import os
import subprocess
from typing import Iterator

import av
import cv2
import decord
import mmcv
import numpy as np
from PIL import Image
import torch
import torchvision
from moviepy.editor import VideoFileClip
from torchvideo.transforms.transforms.transform import Transform
from torchvideo.transforms.transforms.types import PILVideo
from torchvision.transforms import functional as F

from parameters import path_to_ffprobe


def create_image_from_frames(frames, max_frames_per_row=10, margin=2, max_frames=200):
    # Convert frames tensor to NumPy array
    frames_np = frames.numpy() * 255

    # Discard frames if there are too many
    if frames_np.shape[1] > max_frames:
        frames_np = frames_np[:, :max_frames, :, :]

    # Get dimensions
    C, T, H, W = frames_np.shape

    # Calculate total number of rows
    total_rows = (T + max_frames_per_row - 1) // max_frames_per_row

    # Calculate output image dimensions
    output_height = total_rows * (H + margin) + margin
    output_width = min(T, max_frames_per_row) * (W + margin) + margin

    # Create output image
    output_image = np.ones((output_height, output_width, C), dtype=np.uint8) * 255  # White background

    # Fill output image with frames
    for t in range(T):
        row_index = t // max_frames_per_row
        col_index = t % max_frames_per_row

        start_row = row_index * (H + margin) + margin
        start_col = col_index * (W + margin) + margin

        end_row = start_row + H
        end_col = start_col + W

        output_image[start_row:end_row, start_col:end_col, :] = np.transpose(frames_np[:, t, :, :], (1, 2, 0))

    # Convert NumPy array to PIL image
    pil_image = Image.fromarray(output_image)

    return pil_image


def get_fps(video_path, method='decord'):
    """Used when loading from frames"""
    # First, check if metadata file exists
    metadata_path = video_path.replace('videos', 'frames').replace('.mp4', '_metadata.txt')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = f.read()
        metadata = metadata.split('x')
        fps = int(metadata[2].split('/')[0]) / int(metadata[2].split('/')[1])
    else:
        # If metadata file does not exist, read video to get fps
        if method == 'decord':
            vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
            fps = vr.get_avg_fps()
        elif method == 'cv2':
            video = cv2.VideoCapture(str(video_path))
            fps = video.get(cv2.CAP_PROP_FPS)
        elif method == 'ffprobe':
            cmd = f'{path_to_ffprobe} ' \
                  '-v error ' \
                  '-select_streams v ' \
                  '-of default=noprint_wrappers=1:nokey=1 ' \
                  '-show_entries stream=r_frame_rate ' \
                  f'{video_path}'
            fps_str = subprocess.check_output(cmd, shell=True).decode("utf-8").strip()
            fps_num, fps_den = fps_str.split('/')
            fps = float(fps_num) / float(fps_den)
        elif method == 'mmcv':
            vr = mmcv.VideoReader(video_path)
            fps = vr.fps
        elif method == 'av':
            container = av.open(video_path)
            stream = container.streams.video[0]
            fps = float(stream.average_rate)
        elif method == 'torchvision':
            reader = torchvision.io.VideoReader(video_path, "video")
            metadata = reader.get_metadata()
            fps = metadata['video']['fps'][0]
        else:  # method == 'moviepy':
            clip = VideoFileClip(video_path)
            fps = clip.fps
    return fps


def get_duration(video_path, method='decord'):
    """
    Returns the duration of the video in frames
    """
    # First, check if metadata file exists
    metadata_path = video_path.replace('videos', 'frames').replace('.mp4', '_metadata.txt')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = f.read()
        metadata = metadata.split('x')
        fps = eval(metadata[2])
        duration_seconds = float(metadata[3].replace('\n', ''))
        nframes = int(round(fps * duration_seconds))
    else:
        # If metadata file does not exist, read video to get fps
        if method == 'decord':
            vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
            nframes = len(vr)
        elif method == 'cv2':
            video = cv2.VideoCapture(str(video_path))
            nframes = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        elif method == 'ffprobe':
            result = subprocess.run([f'{path_to_ffprobe}', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                                     'stream=nb_frames', '-of', 'default=nokey=1:noprint_wrappers=1', str(video_path)],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            nframes = int(result.stdout)
        elif method == 'mmcv':
            vr = mmcv.VideoReader(video_path)
            nframes = vr.frame_cnt
        elif method == 'av':
            container = av.open(video_path)
            stream = container.streams.video[0]
            nframes = stream.duration
        elif method == 'torchvision':
            reader = torchvision.io.VideoReader(video_path, "video")
            metadata = reader.get_metadata()
            nframes = int(metadata['video']['duration'][0] * metadata['video']['fps'][0])
        else:  # method == 'moviepy':
            clip = VideoFileClip(str(video_path))
            nframes = int(clip.fps * clip.duration)
        return nframes

    return nframes


def get_proportional_sizes(min_side, video_path=None, method='moviepy', width=None, height=None):
    """
    Used when we want to resize the video to a certain size while keeping the aspect ratio.
    The smallest size gets resized to min_side
    """
    if width is None or height is None:
        if method == 'cv2':
            video = cv2.VideoCapture(str(video_path))
            width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        elif method == 'ffprobe':
            cmd = 'ffprobe ' \
                  '-v error ' \
                  '-select_streams v ' \
                  '-of default=noprint_wrappers=1:nokey=1 ' \
                  '-show_entries stream=width,height ' \
                  f'{video_path}'
            width, height = subprocess.check_output(cmd, shell=True).decode("utf-8").strip().split('\n')
        else:  # method == 'moviepy'
            clip = VideoFileClip(video_path)
            width = clip.w
            height = clip.h

    aspect_ratio = float(width) / float(height)
    if width < height:
        new_width = min_side
        new_height = int(new_width * aspect_ratio)
    else:
        new_height = min_side
        new_width = int(new_height * aspect_ratio)
    return new_width, new_height


def _find_matching_dict(list_of_dicts, d_new):
    for d in list_of_dicts:
        if all(d.get(k) == v for k, v in d_new.items()):
            return d['time_transform'], d['time_load'], d['result_mean'], d['result_shape']
    return None


def is_precomputed(data_path, video_path, combination, results):
    """
    Some parameters are orthogonal, and as such, it does not make sense to recompute the same experiments.
    Here we check if the experiment has already been computed, and if so, we return the results.
    """

    params = {'data_path': data_path, 'video_path': video_path, **combination}

    if params['load_format'] == 'random_frames':  # If load_format == 'random_frames', then any frame_sample_rate works
        params_new = {k: v for k, v in params.items() if k != 'frame_sampling_rate'}
        res = _find_matching_dict(results, params_new)
        if res is not None:
            return res

    else:  # If load_format != 'random_frames', then any num_random_frames works
        params_new = {k: v for k, v in params.items() if k != 'num_random_frames'}
        res = _find_matching_dict(results, params_new)
        if res is not None:
            return res

    if params['load_format'] != 'random_clips':
        params_new = {k: v for k, v in params.items() if k not in ['num_random_segments', 'random_segment_duration',
                                                                   'random_segment_before_fps']}
        res = _find_matching_dict(results, params_new)
        if res is not None:
            return res

    elif params['frame_sample_rate'] == -1:  # If frame_sample_rate == -1, then any random_segment_before_fps works.
        params_new = {k: v for k, v in params.items() if k != 'random_segment_before_fps'}
        res = _find_matching_dict(results, params_new)
        if res is not None:
            return res

    # If resize is False, then any short_side_size works. And also any keep_aspect_ratio works.
    if not params['resize']:
        params_new = {k: v for k, v in params.items() if k not in ['short_side_size', 'keep_aspect_ratio']}
        res = _find_matching_dict(results, params_new)
        if res is not None:
            return res

    return None


class NotPossibleException(Exception):
    """
    Some configurations are not possible for certain loaders. In that case, we raise this exception, to distinguish it
    from the case where it is possible to implement, and it is just NotImplemented.
    """
    pass


class PILVideoToTensor(Transform[PILVideo, torch.Tensor, None]):
    """
    For some reason, the torchvideo.transforms.PILVideoToTensor takes forever and explodes the CPU usage. I replace
    the implementation
    """

    def __init__(self, rescale: bool = True, ordering: str = "CTHW"):
        """
        Args:
            rescale: Whether or not to rescale video from :math:`[0, 255]` to
                :math:`[0, 1]`. If ``False`` the tensor will be in range
                :math:`[0, 255]`.
            ordering: What channel ordering to convert the tensor to. Either `'CTHW'`
                or `'TCHW'`
        """
        self.rescale = rescale
        self.ordering = ordering.upper()
        acceptable_ordering = ["CTHW", "TCHW"]
        if self.ordering not in acceptable_ordering:
            raise ValueError(
                "Ordering must be one of {} but was {}".format(
                    acceptable_ordering, self.ordering
                )
            )

    def _gen_params(self, frames: PILVideo) -> None:
        return None

    def _transform(self, frames: PILVideo, params: None) -> torch.Tensor:
        # PIL Images are in the format (H, W, C)
        # np.stack (and posterior creation of a tensor) returns THWC
        if isinstance(frames, Iterator):
            frames = list(frames)
        tensor = torch.tensor(np.stack(frames))
        if self.ordering == "CTHW":
            tensor = tensor.permute(3, 0, 1, 2)
        else:
            tensor = tensor.permute(0, 3, 1, 2)
        if self.rescale:
            tensor = tensor / 255.
        return tensor

    def __repr__(self):
        return (
            self.__class__.__name__
            + "(rescale={rescale!r}, ordering={ordering!r})".format(
                rescale=self.rescale, ordering=self.ordering
            )
        )
