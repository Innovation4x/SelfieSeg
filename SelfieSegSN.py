import cv2
import numpy as np

from models.sn_seg.SINet import *

class SelfieSegSN:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated process.
    """

    def __init__(self, width=320, height=240):
        self.process = None

        # Initialize tflite-interpreter
        self.width = width
        self.height = height

        # Load the sinet pytorch model
        self.config = [[[3, 1], [5, 1]], [[3, 1], [3, 1]],
                  [[3, 1], [5, 1]], [[3, 1], [3, 1]], [[5, 1], [3, 2]], [[5, 2], [3, 4]],
                  [[3, 1], [3, 1]], [[5, 1], [5, 1]], [[3, 2], [3, 4]], [[3, 1], [5, 2]]]
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = SINet(classes=2, p=2, q=8, config=self.config, chnn=1)
        self.model.load_state_dict(torch.load('./models/sn_seg/model_296.pth', map_location=self.device))
        self.model.eval()

        # Enable gpu mode, if cuda available
        self.model.to(self.device)

        # Mean and std. deviation for normalization
        self.mean = [102.890434, 111.25247, 126.91212]
        self.std = [62.93292, 62.82138, 66.355705]
        self.dim = 320

    def seg(self, frame):
        img = np.array(frame)
        img = cv2.resize(img, (self.dim, self.dim))
        img = img.astype(np.float32)

        # Normalize and add batch dimension
        img = (img - self.mean) / self.std
        img /= 255
        img = img.transpose((2, 0, 1))
        img = img[np.newaxis, ...]

        # Load the inputs into GPU
        inps = torch.from_numpy(img).float().to(self.device)

        # Perform prediction and plot results
        with torch.no_grad():
            torch_res = self.model(inps)
            _, mask = torch.max(torch_res, 1)

        # Alpha blending with background image
        mask = mask.view(self.dim, self.dim, 1).cpu().numpy()
        mask = mask * 255
        mask = mask.reshape(self.dim, self.dim).astype("uint8")
        mask = cv2.resize(mask, (self.width, self.height))
        _, mask = cv2.threshold(mask, 128, 255, 0)

        return mask


if __name__ == "__main__":
    width = 320
    height = 240
    seg = SelfieSegSN(width, height)

    # Capture video from camera
    cap = cv2.VideoCapture(0)
    cap.set(3, width)
    cap.set(4, height)

    # Load and resize the background image
    bgd = cv2.imread('./images/whitehouse.jpeg')
    bgd = cv2.resize(bgd, (width, height))

    while cv2.waitKey(1) < 0:
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

        # Show output in window
        cv2.imshow('Selfie Segmentation', out)

    cv2.destroyAllWindows()
