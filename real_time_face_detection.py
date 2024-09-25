"""Real-Time Face Detection
"""
import cv2
import pickle
import os
from PIL import Image
import numpy as np

def real_time_face_detection_using_haar_cascade():
    haar_file = 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(haar_file)

    # define a video capture object
    vid = cv2.VideoCapture(0)

    while(True):

        # Capture the video frame
        # by frame
        ret, frame = vid.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, 'Face Detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()

# Reference : https://github.com/RitvikDayal/Face-Recognition-LBPH/blob/master/Face%20Recognition.ipynb
def real_time_face_detection_using_lbph():
    haar_file = 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(haar_file)

    # define a video capture object
    vid = cv2.VideoCapture(0)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    font = cv2.FONT_HERSHEY_SIMPLEX
    with open('names.pkl', 'rb') as f:
        names = pickle.load(f)
    while True:
        ret, frame = vid.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
             # Check if confidence is less them 100 ==> "0" is perfect match
            if (confidence < 100):
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))

            cv2.putText(frame, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
            cv2.putText(frame, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()

def sample_face_detection():
    names = []
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    name = input('Enter name for the Face: ')
    names.append(name)
    id = names.index(name)
    vid = cv2.VideoCapture(0)
    print('''\n
    Look in the camera Face Sampling has started!.
    Try to move your face and change expression for better face memory registration.\n
    ''')
    # Initialize individual sampling face count
    count = 0

    while(True):
        ret, img = vid.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:

            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
            count += 1

            # Save the captured image into the datasets folder
            cv2.imwrite("dataset/"+name+"." + str(id) + '.' + str(count) + ".jpg", gray[y:y+h,x:x+w])

            cv2.imshow('image', img)

        k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break
        elif count >= 80: # Take 80 face sample and stop video
             break

    with open('names.pkl', 'wb') as f:
        pickle.dump(names, f)

    # Do a bit of cleanup
    print("Your Face has been registered as {}\n\nFace sampling done".format(name.upper()))
    vid.release()
    cv2.destroyAllWindows()

def learning_face_samples():
    # Path for face image database
    path = 'dataset'

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # function to get the images and label data
    def getImagesAndLabels(path):

        image_paths = [os.path.join(path,f) for f in os.listdir(path)]
        faceSamples=[]
        ids = []

        for image_path in image_paths:

            PIL_img = Image.open(image_path).convert('L') # convert it to grayscale
            img_numpy = np.array(PIL_img,'uint8')

            id = int(os.path.split(image_path)[-1].split(".")[1])
            faces = face_detector.detectMultiScale(img_numpy)

            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)

        return faceSamples,ids
    print ("\nTraining for the faces has been started. It might take a while.\n")
    faces,ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))

    # Save the model into trainer/trainer.yml
    recognizer.write('trainer/trainer.yml')

    # Print the numer of faces trained and end program
    print("{0} faces trained. Exiting Training Program".format(len(np.unique(ids))))


def load_model():
    prototxt_path = "weights/deploy.prototxt.txt"
    model_path = "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"
    model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    return model


# Reference : https://thepythoncode.com/article/detect-faces-opencv-python
def real_time_face_detection_using_dnn():
    model = load_model()
    # define a video capture object
    vid = cv2.VideoCapture(0)
    while True:
        ret, frame = vid.read()
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        model.setInput(blob)
        output = np.squeeze(model.forward())
        font_scale = 1.0
        for i in range(0, output.shape[0]):
            # get the confidence
            confidence = output[i, 2]
            # if confidence is above 50%, then draw the surrounding box
            if confidence > 0.5:
                # get the surrounding box cordinates and upscale them to original image
                box = output[i, 3:7] * np.array([w, h, w, h])
                # convert to integers
                top_x, top_y, bottom_x, bottom_y = box.astype(np.int32)
                # draw the rectangle surrounding the face
                cv2.rectangle(frame, (top_x, top_y), (bottom_x, bottom_y), color=(255, 1, 0), thickness=1)
                # draw text as well
                cv2.putText(frame, f"{confidence*100:.2f}%", (top_x, top_y-5), cv2.FONT_HERSHEY_COMPLEX, font_scale, (255, 0, 0), 2)
        # Display the resulting frame
        cv2.imshow('frame', frame)

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()


def main():
    '''
    Driver method
    '''
    print("Choose an option for Real-Time Face Detection:")
    print("1: Haar Cascades")
    print("2: Local Binary Patterns Histograms (LBPH)")
    print("3: Deep Neural Network(DNN)")
    print("4: Exit")
    # Add more options here if needed

    option = input("Enter option number: ")

    if option == '1':
        # Call the function for real-time face detection using Haar Cascade
        real_time_face_detection_using_haar_cascade()
    elif option == '2':
        # Call the function for real-time face detection using LBPH
        sample_face_detection()
        learning_face_samples()
        real_time_face_detection_using_lbph()
    elif option == '3':
        # Call the function for real-time face detection using DNN
        real_time_face_detection_using_dnn()
    elif option == '4':
        exit()

if __name__ == "__main__":
    while(True):
        main()