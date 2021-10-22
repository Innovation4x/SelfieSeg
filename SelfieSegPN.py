
import cv2
import sys, time
import numpy as np
import tensorflow as tf


class SelfieSegPN:
    def __init__(self, width=320, height=240):
        # Initialize tflite-interpreter
        self.width = width
        self.height = height

        self.interpreter = tf.lite.Interpreter(model_path="models/pn_seg/portrait_video.tflite")  # Use 'tf.lite' on recent tf versions
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        h, w = self.input_details[0]['shape'][1:3]
        self.dim = w
        self.prev = np.zeros((self.dim, self.dim, 1))

    def normalize(self, imgOri, scale=1, mean=[103.94, 116.78, 123.68], val=[0.017, 0.017, 0.017]):
        img = np.array(imgOri.copy(), np.float32) / scale
        return (img - mean) * val

    def seg(self, frame):
        img = np.array(frame)
        img = cv2.resize(img, (self.dim, self.dim))
        img = img.astype(np.float32)

        img = self.normalize(img)

        # Add prior as fourth channel
        img = np.dstack([img, self.prev])
        img = img[np.newaxis, :, :, :]

        # Invoke interpreter for inference
        self.interpreter.set_tensor(self.input_details[0]['index'], np.array(img, dtype=np.float32))
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self.output_details[0]['index'])
        out = out.reshape(self.dim, self.dim, 1)
        out = (255 * out).astype("uint8")
        _, out = cv2.threshold(out, 128, 255, cv2.THRESH_BINARY)
        self.prev = (out / 255.0).astype("float32")

        mask = cv2.resize(out, (self.width, self.height))
        _, mask = cv2.threshold(mask, 128, 255, 0)

        return mask


if __name__ == "__main__":
    width = 320
    height = 240
    seg = SelfieSegPN(width, height)

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
