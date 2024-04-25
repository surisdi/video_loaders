import glob
import os
import time
from itertools import product

import numpy as np
import pandas as pd
from tqdm import tqdm

import utils
from loaders import get_loader
from parameters import parameters
from utils import NotPossibleException, create_image_from_frames


def main():
    # all_params contains all the parameters except for data_path
    all_params = {k: v for k, v in parameters.items() if k != 'data_paths'}
    all_combinations = product(*list(all_params.values()))
    all_combinations = [dict(zip(list(all_params.keys()), combination)) for combination in all_combinations]

    # Initialize all random frames and segments, if necessary
    max_num_frame = 50000  # This is for videos > 3h, should be enough
    # Create a random list of N numbers sampled between 0 and max_num_frame, without replacement
    random_frames = {n: sorted(np.random.choice(max_num_frame, n, replace=False))
                     for n in all_params['num_random_frames']}

    random_segment_starts = {n: sorted(np.random.choice(max_num_frame, n, replace=False))
                             for n in all_params['num_random_segments']}

    results = []
    for data_path in parameters['data_paths']:
        # Find all files that end in .mp4
        list_videos = glob.glob(data_path + '/videos/*.mp4', recursive=False)
        for combination in tqdm(all_combinations, desc='data_path: ' + data_path):
            list_random_frames = random_frames[combination['num_random_frames']]
            list_random_segment_starts = random_segment_starts[combination['num_random_segments']]
            for video_path in list_videos:
                precomputed_result = utils.is_precomputed(data_path, video_path, combination, results)
                if precomputed_result is not None:
                    time_transform, time_load, result_mean, result_shape = precomputed_result
                else:
                    try:
                        loader = get_loader(**combination, data_path=data_path, list_random_frames=list_random_frames,
                                            list_random_segment_starts=list_random_segment_starts)
                        
                        time_before = time.time()
                        result_video, result_audio = loader.read_video(video_path)
                        time_after = time.time()
                        """The result_transform contains a (channels, num_frames, height, width) tensor. If the loading 
                        format (access pattern) is random_frames or random_segments, the different frames or segments 
                        are concatenated in the second dimension, so the final format is the same independently of the
                        access pattern."""
                        result_transform = loader.transform(result_video)
                        time_after_transform = time.time()
                        time_load = time_after - time_before
                        time_transform = time_after_transform - time_after

                        result_mean = result_transform.mean()
                        result_shape = list(result_transform.shape)

                    except NotPossibleException as e:
                        time_transform = time_load = result_mean = result_shape = result_transform = None

                    except NotImplementedError as e:
                        time_transform = time_load = result_mean = result_shape = result_transform = None

                result = {
                    'data_path': data_path,
                    'video_path': video_path,
                    'result_mean': result_mean,
                    'result_shape': result_shape,
                    'time_load': time_load,
                    'time_transform': time_transform,
                    'final_result': result_transform
                }
                result = {**result, **combination}

                results.append(result)

    # Reformat results
    reformat_results = {}
    for r in results:
        for key, value in r.items():
            if key not in reformat_results:
                reformat_results[key] = [value]
            else:
                reformat_results[key].append(value)
    videos = reformat_results.pop('final_result')

    df = pd.DataFrame(reformat_results)

    # Save results
    os.makedirs('outputs', exist_ok=True)
    df.to_pickle('outputs/data.pkl')

    # ---------- Reporting and visualization ---------- #
    # Next, we show some examples of how to report and visualize the results. This is just an example, and you can
    # adapt it to your needs.

    # For every 'final_result' in df, create an image that is the concatenation of all the frames, and save that
    # image in the outputs directory, with the name of the rest of parameters
    for index, row in df.iterrows():
        if row['normalize']:
            continue
        video = videos[index]
        if video is not None:
            image_to_save = create_image_from_frames(video)
            format_name = {'random_frames': 'frames', 'random_segments': 'seg', 'all_video': 'all'}[row['load_format']]
            image_name = f"{row['loader_name']}_cc{row['center_crop']}_r{row['resize']}_" + \
                f"a{row['keep_aspect_ratio']}_f{format_name}.jpg"
            os.makedirs(os.path.join('outputs', 'viz'), exist_ok=True)
            image_to_save.save(os.path.join('outputs', 'viz', image_name))

    # Condition on load_format='random_frames' and resize=True
    result_df = df[(df['load_format'] == 'random_frames') & (df['resize'] == True)]

    # Compute the mean of parameter 'time_load' for each value of 'loader_name'
    print(df.groupby('loader_name')['time_load'].mean())  # The mean ignores None values


if __name__ == '__main__':
    main()
