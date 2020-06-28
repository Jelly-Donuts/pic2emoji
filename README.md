pic2emoji-py
============

A Python script for converting images and videos to emojis.

Getting Started
---------------

This script requires at least Python3.6 to run.

### Installation

1. Clone this repository:
   `git clone https://https://github.com/Jelly-Donuts/pic2emoji-py`

2. Within the this project's folder, to install Python dependencies run 
   `python3 -m pip install -r requirements.txt`

3. There is an option to process videos in parallel with the `--parallel` flag.
   This option requires `ffmpeg` to run. Look up the best way to get ffmpeg on
   your machine.

### Usage

To run pic2emoji on an image, run the following command

```
./pic2emoji.py image_path output_path
```

where `image_path` is the path to your input image file and `output_path` is
the path to the desired output file that will be created. You can fine tune
the look of your image (or video) with provided options to choose your 
emoji size and scale your images. Check out [Options](#options) to read more.

To run pic2emoji on video, run

```
./pic2emoji.py video_path output_path --video
```

where `video_path` is the path to your input video file and `output_path` is a
path to the desired output mp4 file that will be created. It is not 
recommended to run pic2emoji on large videos without the `--parallel` option, 
discussed further in the [Options](#options) section below.


Examples
--------

### Image Example

A stock photo (could not find source attribution for the image):

![Stock photo](/examples/stock_photo.jpg)

Size 16px:

![Stock photo emojis 16px](/examples/stock_photo_16px.png)

Size 32px:

![Stock photo emojis 16px](/examples/stock_photo_32px.png)

Size 64px:

![Stock photo emojis 16px](/examples/stock_photo_64px.png)

### Video Example

A clip of the animated short Big Buck Bunny,
[bigbuckbunny.org](https://peach.blender.org/) (licensed under the Creative 
Commons Attribution license):

![Big Buck Bunny clip](/examples/bigbuckbunny_clip.gif)

Size 16px:

![Big Buck Bunny emojis 16px](/examples/bigbuckbunny_clip_16px.gif)

Size 32px:

![Big Buck Bunny emojis 32px](/examples/bigbuckbunny_clip_32px.gif)


Options
-------

There are several options available for configuring the behavior of pic2emoji.

- `--size {16, 32, 64}`: the size of the emoji you'd like to use to 
  tile your image (either 16x16, 32x32, or 64x64). Default is 16. The smaller 
  the emoji size you use, the more emojis you'll have in your image, so play 
  around with what works.

- `--scale SCALE`: the factor by which to scale your image or video before 
  emojifying it. Be aware that the output image or video created will have 
  dimensions `original_width * SCALE, original_height * SCALE`, so you can 
  accidentally created very huge files. SCALE can be any floating point number.

- `--parallel`: process frames of a video file in parallel. This option 
  requires that you have `ffmpeg` installed. It is **highly recommended** to 
  use this flag for videos longer than several seconds, though it is not 
  necessary, you'll just have to be patient.

Notes
-----

- Currently for images, emojis are pasted with their transparency, meaning 
  most of the background will be transparent. You can fix that in the photo-
  editing software of your choice. We may be developing support for different
  background behavior.

  For videos, the transparent pixels are written as black.

- If you want to process a really long video, consider using a larger emoji 
  size, since each increase in size will reduce the processing time by about 
  half! For more fine grained control, use a combination of different emoji 
  sizes and scaling.

