"""
Dict of parameters such that each parameter is a list of values to try. All possible combinations of parameters will be
tried. If specific combinations of parameters are not possible, they will just not be executed.

Some combinations do not affect others. For example, changing frame rate does not affect if we are loading random
frames. Therefore, we do not have to compute the 'random_frames' case for every option in 'frame_sample_rate'. This is
controlled in utils.is_precomputed(). If the experiment has already been computed, the results are copied.
"""

parameters = {
    # Data paths where the videos are located. May be more than one in case you want to compare different disks.
    'data_paths': ['/path/to/data'],
    # Library to use to load the videos.
    'loader_name': ['decord_video', 'dali'],
    # Access pattern to load frames in the video. Options: 'random_frames', 'random_segments', 'all_video'
    'load_format': ['random_frames', 'random_segments'],
    # List with the number of frames to load. The same sampled frames will be used for all loaders.
    'num_random_frames': [5],
    # List with the number of segments to load. The same sampled segments will be used for all loaders.
    'num_random_segments': [3],
    # List with the duration of the segments, in frames (at the returned fps 'frame_sample_rate').
    'random_segment_duration': [12],
    # Whether the initial frame of the segment is sampled before (True) or after (False) considering the final fps.
    # Regardless, random_segment_duration is the duration of the segment (in frames) at the loaded fps.
    'random_segment_before_fps': [True],
    # Resize the video to this size. Options: True, False
    'resize': [False],
    # Resize the video to this size. If 'keep_aspect_ratio' is True, only the shortest side will be resized to this
    # size. Options: any positive integer
    'short_side_size': [224],
    # Options: True, False
    'keep_aspect_ratio': [True],
    # Additionally (after resizing, if 'resize' is True) crop the video to this size. Options: True, False
    'center_crop': [False],
    # Options: any positive integer
    'crop_size': [150],
    # Options: True, False
    'normalize': [False],
    # Sampling rate. Options: any positive integer. If -1, the sampling rate will be the same as the original video.
    'frame_sample_rate': [8],
    # When the loader allows it, the audio is loaded with the loader. If the loader does not allow it, a default audio
    # loader is used to load audio from a pre-extracted .wav file. Note that the pre-processing + storage would be an
    # overhead in this case. This repository does not explore different audio loaders. Options: True, False
    'load_audio': [False],
}

list_loaders = ['decord_video', 'decord_video_gpu', 'pillow', 'pillow_simd', 'opencv_image', 'opencv_video', 
                'mmcv_video', 'mmcv_image', 'pims_pyav', 'pims_imageio', 'pims_moviepy', 'ffmpeg', 'moviepy', 
                'pytorchvideo_frames', 'pytorchvideo_pyav', 'pytorchvideo_torchvision', 'pytorchvideo_decord', 
                'torchvision_videoloader', 'torchvision_videoloader_pyav', 'torchvision_videoloader_cuda', 
                'torchvision_readvideo', 'torchvideo_gulp', 'torchvideo_pil', 'torchvideo_video', 'dali']
assert [loader_name in list_loaders for loader_name in parameters['loader_name']]
assert [load_format in ['random_frames', 'random_segments', 'all_video'] for load_format in parameters['load_format']]


path_to_ffmpeg = 'ffmpeg'  # Sometimes, the full path is needed. For example, '/usr/bin/ffmpeg'
path_to_ffprobe = 'ffprobe'