import cv2
import os.path as osp
import os
import argparse




def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='video.avi')
    parser.add_argument('--input', default='./data/')

    args = parser.parse_args()

    if not os.path.exists(args.out):
        os.makedirs(args.out)

    print("start")
    vidcap = cv2.VideoCapture(args.input)

    print("video capture is done")

    success,image = vidcap.read()
    count = 0
    while success:
      cv2.imwrite(osp.join(args.out, "%d.jpg" % count), image)     # save frame as JPEG file
      success,image = vidcap.read()
      print('Read a new frame: ', success)
      count += 1


if __name__ == "__main__":
    main()