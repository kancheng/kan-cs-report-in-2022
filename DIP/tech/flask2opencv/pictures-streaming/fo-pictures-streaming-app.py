#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from flask import Flask, render_template, Response
import time

app = Flask(__name__)
frames = [open(f + '.png', 'rb').read() for f in ['1', '2', '3', '4', '5']]

def gen_frames():
    counter = 0
    while True:
        n = counter % 5
        print(str(n))
        frame = frames[counter % 5]
        counter += 1
        yield (b'--frame\r\n'
                b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.5)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run('0.0.0.0')