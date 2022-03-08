#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from flask import Flask, render_template, Response
import cv2
import numpy as np

# 找人臉
face = cv2.CascadeClassifier('face/haarcascade_frontalface_default.xml')

# 攝像頭
camera = cv2.VideoCapture(0)

app = Flask(__name__)

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # 轉換成灰度
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 人臉檢測
            faces = face.detectMultiScale(
                gray,     
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(32, 32)
            )
            # 畫方框
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run()

