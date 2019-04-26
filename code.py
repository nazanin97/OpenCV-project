import cv2
import numpy as np
import argparse
from PIL import Image

frame_array = []
all_pics = []


class Pics:
    def __init__(self, x, y):
        self.x = x
        self.y = y


bubble = Image.open("bubble.png")
snow = Image.open("snow.png")
bubble = bubble.resize((10, 10))
snow = snow.resize((10, 10))
# choose an effect for applying on your frames
overlay = bubble


def func2(img, img_gray, mode):

    im = Image.fromarray(img)
    im_g = Image.fromarray(img_gray)
    R, G, B = im_g.convert('RGB').split()
    r = R.load()
    g = G.load()
    b = B.load()
    w, h = im.size
    i = 5
    while i < w:
        j = 5
        if r[i, j] == 0 or g[i, j] == 0 or b[i, j] == 0:
            all_pics.append(Pics(i, j))
        i += 15

    for k in range(len(all_pics)):
        im.paste(overlay, (all_pics[k].x, all_pics[k].y), overlay)

    k = 0
    while k < len(all_pics):
        m = all_pics[k].x
        n = all_pics[k].y
        tmp = 0
        if n+60 < h:
            n += 30
            tmp = 30

        if r[m, n] == 0 or g[m, n] == 0 or b[m, n] == 0:
            all_pics[k].y += tmp

        else:
            all_pics.pop(k)
            k -= 1

        k += 1

    open_cv_image = np.array(im)
    if mode == 0:
        cv2.imshow('test', open_cv_image)
    else:
        cv2.imshow('t', open_cv_image)
        frame_array.append(open_cv_image)


# there is 2 algorithms for extraction KNN and MOG2
def fore_ground_extraction(name):

    mode = 1
    parser = argparse.ArgumentParser(description='How to use background subtraction methods provided by OpenCV.')

    if name != '0':
        parser.add_argument('--input', type=str, help='Path to a source', default=name)

    parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2)', default='MOG2')
    args = parser.parse_args()

    if args.algo == 'MOG2':
        back_sub = cv2.createBackgroundSubtractorMOG2()
    else:
        back_sub = cv2.createBackgroundSubtractorKNN()

    if name != '0':
        capture = cv2.VideoCapture(cv2.samples.findFileOrKeep(args.input))

    else:
        mode = 0
        capture = cv2.VideoCapture(0)

    if not capture.isOpened:
        exit(0)

    while True:

        ret, frame = capture.read()
        if frame is None:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #ret, thresh = cv2.threshold(frame_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        thresh = back_sub.apply(frame)

        func2(frame, thresh, mode)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()

# add effect to video frames


# extract background from foreground (type your video name in the nexet line)
fore_ground_extraction('sample.mp4')

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("output.avi", fourcc, 15, (frame_array[0].shape[1], frame_array[0].shape[0]))

for p in range(len(frame_array)):
    out.write(frame_array[p])

out.release()

# to get online frames with effect uncomment next line
fore_ground_extraction('0')
