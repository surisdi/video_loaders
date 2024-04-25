"""
Pre-process video data.

This script is meant to be run for a limited number of videos (for benchmarking) and it is not optimized for speed.

First, update the paths of the data in parameters.py ('data_paths'), as well as the ffmpeg and ffprobe executables,
if needed. Also set the gulp_videos variable to True if you want to use the TorchVideoGULP loader.

Then, add all the videos you want to process in a 'videos' folder inside each data_path.

Finally, run this script to pre-process the videos. After running the script, the structure of the data folder will be
as follows:

-- /path/to/data
    |-- videos
    |   |-- video1.mkv (or any other format the original video is in)
    |   |-- video1.mp4
    |   |-- video2.mp4
    |   '-- ...
    |
    |-- frames
    |   |-- video1_metadata.txt
    |   |-- video2_metadata.txt
    |   '-- ...
    |   |-- video1
    |   |   |-- 0000000.png
    |   |   |-- 0000001.png
    |   |   |-- ...
    |   |
    |   |-- video2
    |   |   |-- 0000000.png
    |   |   |-- 0000001.png
    |   |   '-- ...
    |   |
    |   '-- ...
    |
    |-- audios
    |   |-- video1.wav
    |   |-- video2.wav
    |   '-- ...
    |
    '-- gulp
        |-- data_0.gulp
        |-- data_1.gulp
        |-- ...
        |-- meta_0.gmeta
        |-- meta_1.gmeta
        '-- ...
"""


import os

from parameters import parameters, path_to_ffmpeg, path_to_ffprobe

os.environ['PATH'] = f"{'/'.join(path_to_ffmpeg.split('/')[:-1])}:{os.environ['PATH']}"

gulp_videos = False  # Only set to True if you are planning on using the TorchVideoGULP loader

if gulp_videos:
    import gulpio
    from gulpio.adapters import AbstractDatasetAdapter
    from gulpio.fileio import GulpIngestor


    class MyGulpAdapter(AbstractDatasetAdapter):
        """
        See examples in https://github.com/achaiah/GulpIO/blob/master/src/main/python/gulpio/adapters.py
        """
        def __init__(self, data_path, video_files, frame_size=-1, shm_dir_path='/dev/shm'):
            self.data_path = data_path
            self.video_files = video_files
            self.frame_size = frame_size
            self.shm_dir_path = shm_dir_path

        def __len__(self):
            return len(self.video_files)

        def get_bursted_frames(self, vid_file):
            with gulpio.utils.temp_dir_for_bursting(self.shm_dir_path) as temp_burst_dir:
                frame_paths = gulpio.utils.burst_video_into_frames(vid_file, temp_burst_dir)
                frames = list(gulpio.utils.resize_images(frame_paths, self.frame_size))
            return frames

        def iter_data(self, slice_element=None):
            for i, vid_file in enumerate(self.video_files):
                frames = self.get_bursted_frames(f'{self.data_path}/videos/{vid_file}')
                vid_id = vid_file.split('.')[0]
                meta = [{'id': i, 'label': ''}]  # Here we would add the correct label if doing classification
                result = {'meta': meta,
                          'frames': frames,
                          'id': vid_id}
                yield result


def main():
    for data_path in parameters['data_paths']:
        # Create directories if they don't exist
        os.makedirs(f'{data_path}/frames', exist_ok=True)
        os.makedirs(f'{data_path}/audios', exist_ok=True)

        # Get list of all video files
        video_files = os.listdir(f'{data_path}/videos/')

        # Process each video file
        for video in video_files:
        
            # Get the video file name and its extension
            video_name, video_ext = os.path.splitext(video)
        
            # 1 - Transform to .mp4 if necessary
            if video_ext != '.mp4':
                os.system(f'{path_to_ffmpeg} '
                          f'-i {data_path}/videos/{video} '
                          '-an -vcodec libx264 -crf 23 '
                          f'{data_path}/videos/{video_name}.mp4')
        
            # 2 - Extract frames
            os.makedirs(f'{data_path}/frames/{video_name}', exist_ok=True)
            os.system(f'{path_to_ffmpeg} '
                      f'-i {data_path}/videos/{video_name}.mp4 '
                      f'-start_number 0 '
                      f'{data_path}/frames/{video_name}/%07d.png')  # Alternatively, you may want to store as jpg
            # Optionally, store metadata such as fps, duration, etc.
            os.system(f'{path_to_ffprobe} '
                      '-v error '  # Suppress warnings (only show errors)
                      '-select_streams v:0 '
                      '-show_entries stream=width,height,duration,r_frame_rate '
                      f'-of csv=s=x:p=0 {data_path}/videos/{video_name}.mp4 '
                      f'> {data_path}/frames/{video_name}_metadata.txt')
        
            # 3 - Extract audio
            os.system(f'{path_to_ffmpeg} '
                      f'-i {data_path}/videos/{video_name}.mp4 '
                      '-vn '
                      '-ac 1 '
                      f'{data_path}/audios/{video_name}.wav')

        # 4 - "Gulp" the videos
        if gulp_videos:  
            gulp_output_folder = f'{data_path}/gulp'
            os.makedirs(gulp_output_folder, exist_ok=True)
            # See https://github.com/achaiah/GulpIO/blob/master/src/main/scripts/gulp_20bn_json_videos
            adapter = MyGulpAdapter(data_path, video_files)
            ingestor = GulpIngestor(adapter, gulp_output_folder, videos_per_chunk=100, num_workers=10)
            ingestor()



if __name__ == '__main__':
    main()
