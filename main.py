import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN
from PIL import Image
import pickle
import os
from fastapi import FastAPI, UploadFile, File, Request
from sklearn.preprocessing import LabelEncoder, Normalizer
from keras_facenet import FaceNet
from fastapi.templating import Jinja2Templates
import uvicorn
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime
import uvicorn

embedder = FaceNet()
templates = Jinja2Templates(directory="templates")
# Define the SQLAlchemy base model
Base = declarative_base()


# Define the StudentAttendance table
class StudentAttendance(Base):
    __tablename__ = 'student_attendance'
    id = Column(Integer, primary_key=True)
    username = Column(String)
    date_time = Column(DateTime)
    names = Column(String)


# Create the database engine and session
engine = create_engine('sqlite:///attendance.db')  # Replace 'attendance.db' with your actual database URI
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()


app = FastAPI()
UPLOAD_FOLDER = 'uploads'


def extract_face(filename, required_size=(160, 160)):
    # Load image from file
    image = Image.open(filename)
    # Convert to RGB, if needed
    image = image.convert('RGB')
    # Convert to array
    pixels = np.asarray(image)
    # Create the detector, using default weights
    detector = MTCNN()
    # Detect faces in the image
    results = detector.detect_faces(pixels)
    # Extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    # Deal with negative pixel index
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # Extract the face
    face = pixels[y1:y2, x1:x2]
    # Resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array


def detect_faces(path):
    # Load the group picture
    image = cv2.imread(path)

    # Create a LBP face detector
    lbp_detector_path = 'D:\\Attendance Management\\Data\\Misc\\haarcascade_frontalface_default.xml'
    lbp_detector = cv2.CascadeClassifier(lbp_detector_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces using the LBP classifier
    rects = lbp_detector.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(10, 10))

    # Create the "faces" folder if it doesn't exist
    if not os.path.exists('faces'):
        os.makedirs('faces')

    # Loop over the detected faces
    for (i, (x, y, w, h)) in enumerate(rects):
        # Adjust the size of the face rectangle
        factor = 0.2
        x -= int(w * factor)
        y -= int(h * factor)
        w = int(w * (1 + factor * 2))
        h = int(h * (1 + factor * 2))

        # Crop the face from the image
        face = image[y:y + h, x:x + w]

        # Save the cropped face as a file
        filename = f'face_{i}.jpg'
        try:
            cv2.imwrite(os.path.join('faces', filename), face)
        except:
            continue


def load_face_test(dir):
    faces = list()
    # Enumerate files
    for filename in os.listdir(dir):
        path = os.path.join(dir, filename)
        try:
            face = extract_face(path)
        except IndexError:
            continue
        if face is not None:
            faces.append(face)
    return faces


def load_dataset_test(dir):
    # List for faces and labels
    X = list()
    path = dir + '/'
    faces = load_face_test(path)
    print("Loaded {} sample(s)".format(len(faces)))
    X.extend(faces)
    return np.asarray(X)


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse('main.html', {"request": request})


@app.post("/recognize_faces")
async def recognize_faces(file: UploadFile = File(...)):
    # Save the uploaded image temporarily
    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(image_path, "wb") as f:
        f.write(await file.read())

    detect_faces(image_path)

    # Load the model
    model = pickle.load(open('D:\\Attendance Management\\MBA\\mba_model.pickle', 'rb'))

    # Load the test dataset
    testX = load_dataset_test('faces')
    print(testX.shape)

    # Convert each face in the test set into embeddings
    emdTestX = embedder.embeddings(testX)
    emdTestX = np.asarray(emdTestX)
    print(emdTestX.shape)

    in_encoder = Normalizer()
    out_encoder = LabelEncoder()
    out_encoder.classes_ = np.load('D:\\Attendance Management\\MBA\\mba_classes.npy')
    emdTestX_norm = in_encoder.transform(emdTestX)
    yhat_test = model.predict(emdTestX_norm)

    names = []
    for selection in range(testX.shape[0]):
        random_face = testX[selection]
        random_face_emd = emdTestX_norm[selection]
        samples = np.expand_dims(random_face_emd, axis=0)
        yhat_class = model.predict(samples)
        yhat_prob = model.predict_proba(samples)
        class_index = yhat_class[0]
        class_probability = yhat_prob[0, class_index] * 100
        predict_names = out_encoder.inverse_transform(yhat_class)
        names.append(predict_names[0])

    # Store the attendance information in the database
    username = "Manan"  # Replace with the actual username
    date_time = datetime.datetime.now()
    attendance = StudentAttendance(username=username, date_time=date_time, names=",".join(names))
    session.add(attendance)
    session.commit()

    # Delete the temporary image file
    os.remove(image_path)

    return {"names": names}


if __name__ == "__main__":
    uvicorn.run("main:app",port=8000)




