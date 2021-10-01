
import cv2, sys, time
import numpy as np
import tensorflow as tf
from PIL import Image

class SelfieSegMNV2:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated process.
    """

    def __init__(self, width=320, height=240):
        self.process = None

        # Initialize tflite-interpreter
        self.width = width
        self.height = height
        self.interpreter = tf.lite.Interpreter(model_path="models/mnv2_seg/deconv_fin_munet.tflite")
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape'][1:3]

        # Image overlay
        self.overlay = np.zeros((self.input_shape[0], self.input_shape[1], 3), np.uint8)
        self.overlay[:] = (127, 0, 0)

    def seg(self, frame):
        # BGR->RGB, CV2->PIL
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        # Resize image
        image = image.resize(self.input_shape, Image.ANTIALIAS)

        # Normalization
        image = np.asarray(image)
        prepimg = image / 255.0
        prepimg = prepimg[np.newaxis, :, :, :]

        # Segmentation
        self.interpreter.set_tensor(self.input_details[0]['index'], np.array(prepimg, dtype=np.float32))
        self.interpreter.invoke()
        outputs = self.interpreter.get_tensor(self.output_details[0]['index'])

        # Process the output
        output = np.uint8(outputs[0] > 0.5)
        res = np.reshape(output, self.input_shape)
        mask = Image.fromarray(np.uint8(res), mode="P")
        mask = np.array(mask.convert("RGB")) * self.overlay
        mask = cv2.resize(np.asarray(mask), (self.width, self.height), interpolation=cv2.INTER_CUBIC)
        mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
        # frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_CUBIC)

        return mask

if __name__ == "__main__":
    width = 320
    height = 240
    seg = SelfieSegMNV2(width, height)

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
