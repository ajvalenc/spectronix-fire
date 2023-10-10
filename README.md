# Real-Time Fire Detection Models

## Models
The detection models are based on Blob detector and Resnet101 model.

### Resnet101
Fine tuning Resnet101 model with custom dataset (retrains using IMAGENET1K_V1 weights by default).

```bash
python train.py
```

### Blob Detector
The Blob detector model is created using OpenCV originally in Python. It is converted to C++ using OpenCV's native C++ API.

```bash
python blob.py
```

## Build Instructions

```bash
 cmake ..
 cmake --config . --build Release -j nproc
```



