# SelfieSeg
Selfie/Portrait Segmentation Models

## Dependencies

* Tensorflow(>=1.14.0), Python 3
* Keras(>=2.2.4), Kito, Scipy, Dlib
* Opencv(>=3.4), PIL, Matplotlib

## Dataset Links

1. [Portseg_128](https://drive.google.com/file/d/1UBLzvcqvt_fin9Y-48I_-lWQYfYpt_6J/view)
2. [Portrait_256](https://drive.google.com/file/d/1FQHaMrsFyxUv5AtwjfPD0gtmEVFM7w3X/view?usp=sharing)
3. [PFCN](https://1drv.ms/u/s!ApwdOxIIFBH19Ts5EuFd9gVJrKTo)
4. [AISegment](https://datasetsearch.research.google.com/search?query=portrait%20segmentation&docid=O3kWsG%2FOg%2FZspufiAAAAAA%3D%3D)
5. [Baidu_Aug](https://drive.google.com/file/d/1zkh7gAhWwoX1nR5GzTzBziG8tgTKtr73/view?usp=sharing)
6. [Supervisely](https://supervise.ly/explore/projects/supervisely-person-dataset-23304/datasets)
7. [Pascal_Person](https://github.com/PINTO0309/TensorflowLite-UNet/tree/master/data_set/VOCdevkit/person)
8. [Supervisely Portrait](https://www.dropbox.com/s/bv8r5vmjc52a8s6/supervisely_portrait.zip)

Also checkout the datset: [UCF Selfie](https://www.crcv.ucf.edu/data/Selfie)

## Mobile-Unet Architecture

SelfieSegMNV2 and SelfieSegMNV3 uses a upsampling block with **Transpose Convolution** with a **stride of 2** for the **decoder part**.

Additionaly, it uses **dropout** regularization to prevent **overfitting**. It also helps our network to learn more **robust** features during training.

## Selfie Segmentation code

**1. SelfieSegMNV2.py**

Here the **inputs and outputs** are images of size **128x128**. The backbone is **mobilenetv2** with **depth multiplier 0.5** as encoder (feature extractor).

**2. SelfieSegMNV3.py**

Here the **inputs and outputs** are images of size **224x224**. The backbone is **mobilenetv3** with **depth multiplier 0.5** as encoder (feature extractor).

**3. SelfieSegPN.py (PortraitNet)**

The **decoder** module consists of refined residual block with **depthwise convolution** and up-sampling blocks with **transpose convolution**. Also, it uses **elementwise addition** instead of feature concatenation in the decoder part. The encoder of the model is **mobilnetev2** and it uses a  **four channel input**, unlike the ohter models, for leveraging temporal consistency. As a result, the output video segmentaion appears more **stabilized** compared to other models. Also, it was observed that depthwise convolution and elementwise addition in decoder greatly **improves the speed** of the model.

* Dataset: Portrait-mix (PFCN+Baidu+Supervisely)
* Size: 224x224

**4. SelfieSegSN.py (SINet: Extreme Lightweight Portrait Segmentation)**

SINet is an **lightweight** portrait segmentaion dnn architecture for mobile devices. The  model which contains around **86.9 K parameters** is able to run at **100 FPS** on iphone (input size -224) , while maintaining the **accuracy** under an 1% margin from the state-of-the-art portrait segmentation method. The proposed portrait segmentation model conatins two new modules for fast and accurate segmentaion viz. **information blocking decoder structure and spatial squeeze modules**.

1. **Information Blocking Decoder**: It measures the confidence in a low-resolution feature map, and blocks the influence of high-resolution feature maps in
highly confident pixels. This prevents noisy information to ruin already certain areas, and allows the model to focuson regions with high uncertainty.  

2. **Spatial Squeeze Modules**: The S2 module is an efficient multipath network for feature extraction. Existing multi-path structures deal with the various size of long-range dependencies by managing multiple receptive fields. However, this increases latency in real implementations, due to having unsuitable structure with regard to kernel launching and synchronization. To mitigate this problem, they squeeze the spatial resolution from each feature map by average pooling, and show that this is more effective than adopting multi-receptive fields.


Besides the aforementioned features, the SINet architecture uses **depthwise separable convolution and PReLU actiavtion** in the encoder modules. They also use **Squeeze-and-Excitation** (SE) blocks that adaptively recalibrates channel-wise feature responses by explicitly modelling interdependencies between channels, for improving the model accuracy. For training, they used cross entropy loss with additional **boundary refinement**. In general it is **faster and smaller** than most of the  portrait segmentaion models; but in terms of accuracy it falls behind portrait-net model by a small margin. The model seems to be **faster than mobilentv3** in iOS; but in android it seems likely to make only a marginal difference(due to optimized tflite swish operator).


We trained the sinet model with **aisegment + baidu portrait** dataset using input size **320** and cross entropy loss function, for 600 epochs and achieved an **mIOU of  97.5%**. The combined dataset consists of around **80K images**(train+val), after data augmentaion. The final trained model has a size of **480kB** and **86.91K parameters**.

## License

This project is licensed under the terms of the [MIT](LICENSE) license.

## Versioning

Version 1.0

## Acknowledgments
* https://github.com/anilsathyan7/Portrait-Segmentation
* https://www.tensorflow.org/model_optimization
* https://www.tensorflow.org/lite/performance/gpu_advanced
* https://github.com/cainxx/image-segmenter-ios
* https://github.com/gallifilo/final-year-project
* https://github.com/dong-x16/PortraitNet
* https://github.com/ZHKKKe/MODNet
* https://github.com/clovaai/ext_portrait_segmentation
* https://github.com/tantara/JejuNet
* https://github.com/lizhengwei1992/mobile_phone_human_matting
* https://github.com/dailystudio/ml/tree/master/deeplab
* https://github.com/PINTO0309/TensorflowLite-UNet
* https://github.com/xiaochus/MobileNetV3
* https://github.com/yulu/GLtext
* https://github.com/berak/opencv_smallfry/blob/master/java_dnn
* https://github.com/HasnainRaz/SemSegPipeline
* https://github.com/onnx/tensorflow-onnx
* https://github.com/onnx/keras-onnx
* https://machinethink.net/blog/mobilenet-v2/
* [On-Device Neural Net Inference with Mobile GPUs](https://arxiv.org/pdf/1907.01989.pdf)
* [AI Benchmark: All About Deep Learning on Smartphones in 2019](https://arxiv.org/pdf/1910.06663.pdf)
* [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)
* [Google AI Blog: MobilenetV3](https://ai.googleblog.com/2019/11/introducing-next-generation-on-device.html)
* [Youtube Stories: Mobile Real-time Video Segmentation ](https://ai.googleblog.com/2018/03/mobile-real-time-video-segmentation.html)
* [Facebook SparkAR: Background Segmentation](https://sparkar.facebook.com/ar-studio/learn/documentation/tracking-people-and-places/segmentation/)
* [Learning to Predict Depth on the Pixel 3 Phones](https://ai.googleblog.com/2018/11/learning-to-predict-depth-on-pixel-3.html)
* [uDepth: Real-time 3D Depth Sensing on the Pixel 4](https://ai.googleblog.com/2020/04/udepth-real-time-3d-depth-sensing-on.html)
* [iOS Video Depth Maps Tutorial](https://www.raywenderlich.com/5999357-video-depth-maps-tutorial-for-ios-getting-started)
* [Huawei: Portrait Segmentation](https://developer.huawei.com/consumer/en/doc/20201601)
* [Deeplab Image Segmentation](https://colab.research.google.com/github/tensorflow/models/blob/master/research/deeplab/deeplab_demo.ipynb)
* [Tensorflow - Image segmentation](https://www.tensorflow.org/beta/tutorials/images/segmentation)
* [Official Tflite Segmentation Demo](https://github.com/tensorflow/examples/tree/master/lite/examples/image_segmentation)
* [Tensorflowjs - Tutorials](https://www.tensorflow.org/js)
* [Hyperconnect - Tips for fast portrait segmentation](https://hyperconnect.github.io/2018/07/06/tips-for-building-fast-portrait-segmentation-network-with-tensorflow-lite.html)
* [Prismal Labs: Real-time Portrait Segmentation on Smartphones](https://blog.prismalabs.ai/real-time-portrait-segmentation-on-smartphones-39c84f1b9e66)
* [Keras Documentation](https://keras.io/)
* [Boundary-Aware Network for Fast and High-Accuracy Portrait Segmentation](https://arxiv.org/pdf/1901.03814.pdf)
* [Fast Deep Matting for Portrait Animation on Mobile Phone](https://arxiv.org/pdf/1707.08289.pdf)
* [Adjust Local Brightness for Image Augmentation](https://medium.com/@fanzongshaoxing/adjust-local-brightness-for-image-augmentation-8111c001059b)
* [Pyimagesearch - Super fast color transfer between images](https://www.pyimagesearch.com/2014/06/30/super-fast-color-transfer-images/)
* [OpenCV with Python Blueprints](https://subscription.packtpub.com/book/application_development/9781785282690/1/ch01lvl1sec11/generating-a-warming-cooling-filter)
* [Pysource - Background Subtraction](https://pysource.com/2018/05/17/background-subtraction-opencv-3-4-with-python-3-tutorial-32/)
* [Learn OpenCV - Seamless Cloning using OpenCV](https://www.learnopencv.com/seamless-cloning-using-opencv-python-cpp/)
* [Deep Image Harmonization](https://github.com/wasidennis/DeepHarmonization)
* [Tfjs Examples - Webcam Transfer Learning](https://github.com/tensorflow/tfjs-examples/blob/fc8646fa87de990a2fc0bab9d1268731186d9f04/webcam-transfer-learning/index.js)
* [Opencv Samples: DNN-Classification](https://github.com/opencv/opencv/blob/master/samples/dnn/classification.py)
* [Deep Learning In OpenCV](https://elinux.org/images/9/9e/Deep-Learning-in-OpenCV-Wu-Zhiwen-Intel.pdf)
* [BodyPix - Person Segmentation in the Browser](https://github.com/tensorflow/tfjs-models/tree/master/body-pix)
* [High-Resolution Network for Photorealistic Style Transfer](https://arxiv.org/pdf/1904.11617.pdf)
* [Tflite Benchmark Tool](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark)
* [TensorFlow Lite Android Support Library](https://github.com/tensorflow/tensorflow/blob/764a3ab93ac7425b49b9c13dc151bc9c2f2badf6/tensorflow/lite/experimental/support/java/README.md)
* [TensorFlow Lite Hexagon delegate](https://www.tensorflow.org/lite/performance/hexagon_delegate)
* [Tensorflow lite gpu delegate inference using opengl and SSBO in android](https://github.com/tensorflow/tensorflow/issues/26297)
* [Udacity: Intel Edge AI Fundamentals Course](https://www.udacity.com/scholarships/intel-edge-ai-scholarship)
* [Udacity: Introduction to TensorFlow Lite](https://www.udacity.com/course/intro-to-tensorflow-lite--ud190)
* [Android: Hair Segmentation with GPU](https://github.com/google/mediapipe/blob/master/mediapipe/docs/examples.md#hair-segmentation-with-gpu)
* [Image Effects for Android using OpenCV: Image Blending](https://heartbeat.fritz.ai/image-effects-for-android-using-opencv-image-blending-319e0e042e27)
* [Converting Bitmap to ByteBuffer (float) in Tensorflow-lite Android](https://stackoverflow.com/questions/55777086/converting-bitmap-to-bytebuffer-float-in-tensorflow-lite-android)
* [Real-time Hair Segmentation and Recoloring on Mobile GPUs](https://arxiv.org/pdf/1907.06740.pdf)
* [PortraitNet: Real-time portrait segmentation network for mobile device](https://www.sciencedirect.com/science/article/abs/pii/S0097849319300305)
* [ONNX2Keras Converter](https://github.com/nerox8664/onnx2keras)
* [Google: Coral AI](https://coral.ai/docs/accelerator/get-started/)
* [Hacking Google Coral Edge TPU](https://towardsdatascience.com/hacking-google-coral-edge-tpu-motion-blur-and-lanczos-resize-9b60ebfaa552)
* [Peter Warden's Blog: How to Quantize Neural Networks with TensorFlow](https://petewarden.com/2016/05/03/how-to-quantize-neural-networks-with-tensorflow/)
* [Tensorflow: Post Training Quantization](https://www.tensorflow.org/lite/performance/post_training_quantization)
* [Qualcomm Hexagon 685 DSP is a Boon for Machine Learning](https://www.xda-developers.com/qualcomm-snapdragon-845-hexagon-685-dsp)
* [How Qualcomm Brought Tremendous Improvements in AI Performance to the Snapdragon 865](https://www.xda-developers.com/qualcomm-snapdragon-865-ai-performance-machine-learning-analysis/)
* [TF-TRT 2.0 Workflow With A SavedModel](https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html#worflow-with-savedmodel)
* [NVIDIA-AI-IOT: Deepstream_Python_Applications](https://github.com/NVIDIA-AI-IOT/deepstream_python_apps)
* [Awesome Tflite: Models, Samples, Tutorials, Tools & Learning Resources.](https://github.com/margaretmz/awesome-tflite)
* [Google: Machine Learning Bootcamp for Mobile Developers](https://www.youtube.com/watch?v=uMokEy_921Q)
* [Machinethink: New mobile neural network architectures](https://machinethink.net/blog/mobile-architectures/)
* [Deeplab Tflite Tfhub](https://tfhub.dev/s?publisher=sayakpaul)
* [MediaPipe with Custom tflite Model](https://blog.gofynd.com/mediapipe-with-custom-tflite-model-d3ea0427b3c1)
* [Google Mediapipe Github](https://github.com/google/mediapipe)
