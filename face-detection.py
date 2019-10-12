import cv2 as cv

WIDTH = 400
HEIGHT = 300
ID_WIDTH = 3
ID_HEIGHT = 4
cap = cv.VideoCapture(0)
cap.set(ID_WIDTH, WIDTH)
cap.set(ID_HEIGHT, HEIGHT)

face_engine = cv.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')
eye_engine = cv.CascadeClassifier('data/haarcascades/haarcascade_eye.xml')
smile_engine = cv.CascadeClassifier('data/haarcascades/haarcascade_smile.xml')


def frames():
    while True:

        ret, frame = cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # 面部检测
        faces = face_engine.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)
        for (x, y, w, h) in faces:
            face_area = gray[y:y + h, x: x + w]
            face_area_color = frame[y:y + h, x: x + w]
            point1 = (x, y)
            point2 = (x + w, y + h)
            color = (255, 0, 0)
            width = 2
            frame = cv.rectangle(frame, point1, point2, color, width)

            # 人眼检测
            eyes = eye_engine.detectMultiScale(face_area, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))
            for (ex, ey, ew, eh) in eyes:
                point1 = (ex, ey)
                point2 = (ex + ew, ey + eh)
                color = (0, 255, 0)
                width = 1
                cv.rectangle(face_area_color, point1, point2, color, width)

            # 微笑检测
            smiles = smile_engine.detectMultiScale(face_area, scaleFactor=1.1, minNeighbors=65, minSize=(40, 40),
                                                   flags=cv.CASCADE_SCALE_IMAGE)
            for (sx, sy, sw, sh) in smiles:
                point1 = (sx, sy)
                point2 = (sx + sw, sy + sh)
                color = (0, 0, 255)
                width = 1
                cv.rectangle(face_area_color, point1, point2, color, width)
                cv.putText(frame, "smile", (x, y - 7), 3, 1.2, (0, 0, 255), 2, cv.LINE_AA)

        cv.imshow('color', frame)
        cv.imshow('gray', gray)
        cv.waitKey(25)


frames()

