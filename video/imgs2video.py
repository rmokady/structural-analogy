import cv2
import numpy as np
import os
import argparse
from os.path import isfile, join


def convert_frames_to_video(args):
    frame_array = []
    #files = [f for f in os.listdir(args.input) if isfile(join(args.input, f))]
    files = []
    for i in range(args.frames):
        if args.const:
            files.append(str(args.start_cnt) + args.ext)
        elif not args.reverse:
            files.append(str(args.start_cnt + i) + args.ext)
        else:
            files.append(str(args.start_cnt + args.frames - i -1) + args.ext)

    print(files)

    # for sorting the file names properly
    #files.sort(key=lambda x: int(x[5:-4]))

    for i in range(len(files)):
        filename = args.input + files[i]
        # reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        print(filename)
        # inserting the frames into an image array
        frame_array.append(img)

    out = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*'DIVX'), args.fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='video.avi')
    parser.add_argument('--input', default='./data/')
    parser.add_argument('--ext', default='.png')
    parser.add_argument('--fps', type=float, default=15.0)
    parser.add_argument('--start_cnt', type=int, default=0)
    parser.add_argument('--frames', type=int, default=10)
    parser.add_argument('--reverse', type=bool, default=False)
    parser.add_argument('--const', type=bool, default=False)

    args = parser.parse_args()

    #convert_frames_to_video(pathIn, pathOut, fps)
    convert_frames_to_video(args)


if __name__ == "__main__":
    main()