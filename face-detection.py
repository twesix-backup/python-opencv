import cv2 as cv

cap = cv.VideoCapture(0)
face_engine = cv.CascadeClassifier('data/haarcascades/haarcascade_frontalface_default.xml')
eye_engine = cv.CascadeClassifier('data/haarcascades/haarcascade_eye.xml')
smile_engine = cv.CascadeClassifier('data/haarcascades/haarcascade_smile.xml')


def frames():
    while True:

        ret, frame = cap.read()

        # 面部检测
        faces = face_engine.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=3)
        for (x, y, w, h) in faces:
            face_area = frame[y:y + h, x: x + w]
            frame = cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # 人眼检测
            eyes = eye_engine.detectMultiScale(face_area)
            for (ex, ey, ew, eh) in eyes:
                cv.rectangle(face_area, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)

            # 微笑检测
            smiles = smile_engine.detectMultiScale(face_area, scaleFactor=1.16, minNeighbors=65, minSize=(25, 25),
                                                   flags=cv.CASCADE_SCALE_IMAGE)
            for (sx, sy, sw, sh) in smiles:
                cv.rectangle(face_area, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 1)
                cv.putText(frame, "smile", (x, y - 7), 3, 1.2, (0, 0, 255), 2, cv.LINE_AA)

            cv.imshow('camera', frame)
            cv.waitKey(10)

    cap.release()
    cv.destroyAllWindows()


frames()

