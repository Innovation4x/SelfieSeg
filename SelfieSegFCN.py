
from PIL import Image
import cv2
import time
import numpy as np
from torchvision import models
import torch
from torchvision import transforms
import portrait_segment_v3 as ps

class SelfieSegFCN:
    def __init__(self, width=320, height=240):
        self.width = width
        self.height = height
        self.imgsize = width
        if width > height:
            self.imgsize = height
        self.dev = "cpu"
        self.dev = "gpu"
        self.model = models.segmentation.fcn_resnet101(pretrained=1).eval()

    def seg(self, frame):
        mask = ps.portait_segment(self.model, frame, self.imgsize, dev=self.dev)
        _, mask = cv2.threshold(mask, 128, 255, 0)

        return mask

if __name__ == "__main__":
    #"""
    width = 480
    height = 360
    seg = SelfieSegFCN(width, height)

    # Capture video from camera
    cap = cv2.VideoCapture(0)
    cap.set(3, width)
    cap.set(4, height)

    # Load and resize the background image
    bgd = cv2.imread('./images/background.jpeg')
    bgd = cv2.resize(bgd, (width, height))

    elapsedTime = 0
    count = 0

    while cv2.waitKey(1) < 0:
        t1 = time.time()

        # Read input frames
        success, frame = cap.read()
        if not success:
           cap.release()
           break

        # Get segmentation mask
        mask = seg.seg(frame)

        # Merge with background
        fg = cv2.bitwise_or(frame, frame, mask=mask)
        bg = cv2.bitwise_or(bgd, bgd, mask=~mask)
        out = cv2.bitwise_or(fg, bg)

        elapsedTime += (time.time() - t1)
        count += 1
        fps = "{:.1f} FPS".format(count / elapsedTime)

        # Show output in window
        cv2.putText(out, fps, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38, 255, 38), 1, cv2.LINE_AA)
        cv2.imshow('Selfie Segmentation', out)

    cv2.destroyAllWindows()
    cap.release()
