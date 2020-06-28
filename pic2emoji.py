#!/usr/bin/env python3
"""
File: pic2emoji.py
------------------
Turn your pictures and videos into emojis!

Usage: ./pic2emoji.py input_path output_path [--video [--parallel]] 
           [--scale SCALE] [--size {16, 32, 64}]

Specify the input file path and the desired output file path. If the input 
file is a video, provide the `--video` flag. You can provide the `--parallel` 
flag as well to process the video in parallel; this option requires that you 
have ffmpeg installed (you can check for installation by running the command
`which ffmpeg`). For very short videos, it is not recommended to use parallel
as the extra overhead overshadows the benefit of parallelization.

The `--scale` option allows you to specify that you want the input image or 
video to be scaled by some factor before being converted to an emoji. Scaling
your input can give higher quality results but will result in larger output 
files.

The `--size` option allow you to specify what size emojis to use to tile the 
result. We present 16, 32, and 64 pixel options, with 16 as the default.

For images, the transparency of emoji backgrounds will be preserved. For video,
all transparent pixels in the emojis will be replaced with black pixels.

This script requires FFMPEG to 

"""

import argparse
from functools import partial
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import subprocess
import time
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors

DIR = os.path.join(Path(__file__).parent, 'data')
DEFAULT_SIZE = 16  # 16x16px emojis
ULIMIT_FILE = 256  # 256 open file limit default on macOS


def main(
    input_path: str,
    output_path: str,
    video: bool = False, 
    scale: float = 1.0,
    size: int = DEFAULT_SIZE,
    parallel: bool = False
) -> None:
    if not video:  # emojify an image
        create_globals(size)
        
        img = cv2.imread(input_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2 reads in BGR
        if scale != 1.0:
            img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

        emoji_img = emojify(img, size)
        emoji_img.save(output_path)

    else:  # emojify a video
        frame_width, frame_height, frame_count, _ = get_video_meta(input_path)
        num_procs = mp.cpu_count() if parallel else 1
        batch_size = frame_count // num_procs
        print(f'Processing video on {num_procs} process(es), each with a batch '
              f'size of {batch_size} frames.')

        proc_fn = partial(proc_emojify_video, video_fname=input_path,
                          batch_size=batch_size, size=size, scale=scale)

        start_time = time.time()

        if num_procs > 1:
            with mp.Pool(num_procs) as pool:
                for i in range(num_procs):
                    pool.apply_async(proc_fn, (i, f'/tmp/pic2emoji_{i}.mp4'))

                pool.close()
                pool.join()

            combine_output_files(num_procs, output_path)
        else:
            proc_fn(0, output_path)

        end_time = time.time()

        proc_time = end_time - start_time
        print(f'Time to process video: {proc_time:.5} sec')
        print(f'FPS: {frame_count / proc_time:.5} frames/sec')
   

def parse_args() -> dict:
    parser = argparse.ArgumentParser(description='Turn images and videos into '
                                                 'emojis.')
    parser.add_argument('input_path', help='path to image or video file')
    parser.add_argument('output_path', help='output file path')
    parser.add_argument('--video', action='store_true', 
                        help='the input is a video.')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='factor to scale the image or video by before '
                             'emojiyfing.')
    parser.add_argument('--size', type=int, default=DEFAULT_SIZE,
                        choices=[16, 32, 64], help='size of desired emojis in '
                                                   'pixels (size x size).')
    parser.add_argument('--parallel', action='store_true',
                        help='parallelize video processing, requires ffmpeg')
    return vars(parser.parse_args())


def create_globals(size: int = DEFAULT_SIZE) -> None:
    """
    Creates global variables for the script based on the emoji size specified.

    Creates:
        - FILES: a list of emoji image file paths.
        - KNN: an sklearn KNN classifier
        - SIZE: the specified emoji size.
        - ALPHA_CHANNEL: a default alpha channel of the correct size for 
              for emojis stored in RGB.

    Args:
        size: the size of the square emoji in pixels.

    """

    global FILES, KNN, SIZE, ALPHA_CHANNEL

    SIZE = size  # options: 16, 32, 64
    emoji_dir = os.path.join(DIR, f'emoji{SIZE}')
    FILES = sorted([os.path.join(emoji_dir, f) for f in os.listdir(emoji_dir)
                    if os.path.isfile(os.path.join(emoji_dir, f))])
    knn_file = os.path.join(DIR, f'knn{SIZE}.pkl')
    emoji_json = os.path.join(DIR, f'emojis{SIZE}.json')
    
    # Load cache
    cache = json.load(open(emoji_json, 'r'))
    avgs = np.array(cache['avg'])
    assert cache['size'] == SIZE, "Cache size doesn't match specified size."
    ALPHA_CHANNEL = np.ones((SIZE, SIZE), dtype=np.uint8)

    # Load KNN model
    if os.path.exists(knn_file):
        KNN = pickle.load(open(knn_file, 'rb'))
    else:
        KNN = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(avgs)
        pickle.dump(KNN, open(knn_file, 'wb'))


def emojify(
    img: np.ndarray,
    size: int = DEFAULT_SIZE,
    alpha: bool = True,
    cache: Optional[dict] = None
) -> Image:
    """
    Turns an input image into an image of emojis.

    Args:
        img: the original image.
        size: the size in pixels of the square emojis.
        knn: the sklearn KNN model.
        alpha: whether to include an alpha channel.
        cache: a provided dictionary of emoji images.
    
    Returns:
        the new emoji pillow image.

    """

    # Find the average of every size x size patch of the image
    avg_patches = np.array([
        [img[i:i + size, j:j + size].mean(axis=(0, 1))
          for j in range(0, img.shape[1], size)]
         for i in range(0, img.shape[0], size)
    ])
    patches_list = avg_patches.reshape(
        (avg_patches.shape[0] * avg_patches.shape[1], 3)
    )
    
    # Find nearest neighbors to each patch
    _, inds = KNN.kneighbors(patches_list)
    inds = inds.reshape((avg_patches.shape[0], avg_patches.shape[1]))
    
    # Create a new blank image
    dims = 'RGBA' if alpha else 'RGB'
    canvas = Image.new(dims, (img.shape[1], img.shape[0])) # change to rgba

    # Load all necessary emoji images before pasting
    if cache is None:
        to_open = np.unique(inds)
        if len(to_open) > ULIMIT_FILE:
            cache = {idx: load_image(idx) for idx in to_open}
        else:  # faster to open directly if possible
            cache = {x: Image.open(FILES[x]) for x in to_open}

    # Paste all the emojis
    for y in range(inds.shape[0]):
        for x in range(inds.shape[1]):
            canvas.paste(cache[inds[y, x]], (x*size, y*size))
    
    return canvas


def proc_emojify_video(
    proc_num: int,
    output_fname: str,
    video_fname: str,
    batch_size: int,
    size: int = DEFAULT_SIZE,
    scale: float = 1.0
) -> None:
    """
    Emojifies part of a video file. To be run by multiple processes.

    Args:
        proc_num: this process' number.
        output_fname: the desired output video filename.
        video_fname: the filename of the input video.
        batch_size: the number of frames for this process to emojify.
        size: the size in pixels of the emoji to use.
        scale: the factor by which to scale the frames.

    """

    # Each process is it's own Python instance
    create_globals(size)

    cap = cv2.VideoCapture(video_fname)
    if cap is None:
        raise Exception(f'Failed to load video "{video_fname}."')

    # Jump to starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, batch_size * proc_num)
    
    width, height, num_frames, fps = get_video_meta(cap)
    
    out = cv2.VideoWriter(
        output_fname,
        cv2.VideoWriter_fourcc(*'avc1'),    # Apple's version of MPEG4
        fps,
        (int(width * scale), int(height * scale))
    )

    # Cache emojis
    cache = list(map(partial(load_image, raw=False), range(len(FILES))))

    for _ in range(batch_size):
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if scale != 1.0:
            frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

        emoji_pil = emojify(frame, SIZE, alpha=False, cache=cache)
        emoji_frame = cv2.cvtColor(np.array(emoji_pil), cv2.COLOR_RGB2BGR)

        out.write(emoji_frame)
    
    cap.release()
    out.release()


def get_video_meta(
    video: Union[str, cv2.VideoCapture]
) -> Tuple[int, int, int]:
    """
    Retrieves video metadata.

    Args:
        video: the video filename or the opened VideoCapture object.

    Returns:
        the frame width, frame height, frame count, and fps.

    """

    should_release = False
    if isinstance(video, str): 
        cap = cv2.VideoCapture(video)
    else:
        cap = video

    if not cap.isOpened():
        raise Exception(f'Could not open video "{video}".')

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    if should_release:
        cap.release()

    return frame_width, frame_height, frame_count, fps


def combine_output_files(num_procs: int, output_fname: str) -> None:
    """
    Combines the partial videos created by multiple processes.

    Args:
        num_procs: the number of processes that processed the video.
        output_fname: the filename of the final output video.

    """

    # Store intermediate output filenames in temporary file
    temp_files = [f'/tmp/pic2emoji_{i}.mp4' for i in range(num_procs)]

    temp_files_txt = '/tmp/temp_files.txt'
    with open(temp_files_txt, 'w') as f:
        for t in temp_files:
            f.write(f'file {t} \n')

    # Combine files using ffmpeg
    ffmpeg_cmd = (f'ffmpeg -y -loglevel error -f concat -safe 0 -i '
                  f'{temp_files_txt} -vcodec copy {output_fname}')

    try:
        subprocess.run(ffmpeg_cmd, shell=True, check=True)
    finally:
        # Remove the temperory output files
        for f in temp_files:
            os.remove(f)
        os.remove(temp_files_txt)


def load_image(idx: int, raw: bool = True) -> Image:
    """
    Load an emoji image at the specified index without keeping the file open.

    This is function provides a mechanism around hitting the open file cap on 
    users' systems (256 on macOS).

    Args:
        idx: the index of the emoji image to load within FILES.
        raw: whether to read the image in as stored, or in RGB mode.

    Returns:
        the emoji pillow image.

    """

    if raw:
        img_arr = cv2.imread(FILES[idx], cv2.IMREAD_UNCHANGED)

        # Grayscale
        if len(img_arr.shape) == 2:
            return Image.fromarray(img_arr, 'L')

        # BGR
        if img_arr.shape[2] == 3:
            return Image.fromarray(cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB))
        
        # BGRA
        return Image.fromarray(cv2.cvtColor(img_arr, cv2.COLOR_BGRA2RGBA))
    
    else:
        img_arr = cv2.imread(FILES[idx])
        return Image.fromarray(cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB))


if __name__ == '__main__':
    main(**parse_args())
