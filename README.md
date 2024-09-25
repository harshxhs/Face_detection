### Real-Time Face Detection

#### Installation Setup
- Open Visual Studio as 'Run as administrator'
```
    python3 -m venv .venv
    source .venv\Scripts\activate
    pip install -r requirements.txt
    python real_time_face_detection.py
```
#### Implemented three algorithms using OpenCV
- Haar Cascades: This is a machine learning object detection method used to identify objects in an image or video. For face detection, it is trained on thousands of positive images (with faces) and negative images (without faces).

- Local Binary Patterns Histograms (LBPH): This is another method for face detection provided by OpenCV. It is considered to be more accurate than Haar cascades but is computationally more expensive.

- Deep Neural Network (DNN) module: OpenCV's DNN module includes a pre-trained model for face detection that is based on the Single Shot Detector (SSD) framework with a ResNet base network. This method is more accurate than Haar cascades and works well for real-time applications.
