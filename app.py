from flask import Flask, request, render_template
from google.cloud import vision
import io
import cv2
import numpy as np
import os

app = Flask(__name__)

# Configura la ruta a tus credenciales de Google Cloud
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'credenciales.json'

ADJUSTMENTS = {
    1: {'x': -3, 'y': 1},  # Extremo derecho ceja izquierda
    2: {'x': -2, 'y': 0},   # Extremo izquierdo ceja izquierda
    3: {'x': 5, 'y': 0},  # Extremo derecho ceja derecha
    4: {'x': -23, 'y': 0}, # Extremo izquierdo ceja derecha
    5: {'x': 2, 'y': 1},  # Centro del ojo izquierdo
    6: {'x': -3, 'y': 2},  # Lado izquierdo ojo izquierdo
    7: {'x': 4, 'y': 4},   # Lado derecho ojo izquierdo
    8: {'x': -2, 'y': 3},   # Centro del ojo derecho
    9: {'x': -3, 'y': 0},  # Lado izquierdo ojo derecho
    10: {'x': 4, 'y': 6},  # Lado derecho ojo derecho
    11: {'x': 3, 'y': 1},  # Punta de la nariz
    12: {'x': -2, 'y': 1},  # Labio superior izquierdo
    13: {'x': -1, 'y': 2},  # Labio superior derecho
    14: {'x': -4, 'y': 3},  # Labio inferior izquierdo
    15: {'x': -2, 'y': 3}   # Labio inferior derecho
}

def detect_face_landmarks(image_path):
    client = vision.ImageAnnotatorClient()
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.face_detection(image=image)
    faces = response.face_annotations
    if not faces:
        return []
    landmarks = faces[0].landmarks
    return landmarks

def adjust_point(point, adjustment):
    return vision.Vertex(
        x=int(point.x) + adjustment['x'],
        y=int(point.y) + adjustment['y']
    )

def resize_image(image_path, output_size):
    img = cv2.imread(image_path)
    resized_img = cv2.resize(img, output_size)
    cv2.imwrite(image_path, resized_img)
    return resized_img

def process_image(image_path):
    img = cv2.imread(image_path)
    original_height, original_width = img.shape[:2]

    if original_width != 96 or original_height != 96:
        img = resize_image(image_path, (96, 96))

    points = detect_face_landmarks(image_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img_colored = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)

    if len(points) > 0:
        desired_landmarks = []
        desired_landmarks.append(adjust_point(points[3].position, ADJUSTMENTS[1]))
        desired_landmarks.append(adjust_point(points[4].position, ADJUSTMENTS[2]))
        desired_landmarks.append(adjust_point(points[5].position, ADJUSTMENTS[3]))
        desired_landmarks.append(adjust_point(points[6].position, ADJUSTMENTS[4]))

        center_left_eye = adjust_point(points[1].position, ADJUSTMENTS[5])
        left_eye_x = int(center_left_eye.x)
        left_eye_y = int(center_left_eye.y)
        desired_landmarks.append(center_left_eye)
        desired_landmarks.append(vision.Vertex(x=left_eye_x + ADJUSTMENTS[6]['x'], y=left_eye_y + ADJUSTMENTS[6]['y']))
        desired_landmarks.append(vision.Vertex(x=left_eye_x + ADJUSTMENTS[7]['x'], y=left_eye_y + ADJUSTMENTS[7]['y']))

        center_right_eye = adjust_point(points[0].position, ADJUSTMENTS[8])
        right_eye_x = int(center_right_eye.x)
        right_eye_y = int(center_right_eye.y)
        desired_landmarks.append(center_right_eye)
        desired_landmarks.append(vision.Vertex(x=right_eye_x + ADJUSTMENTS[9]['x'], y=right_eye_y + ADJUSTMENTS[9]['y']))
        desired_landmarks.append(vision.Vertex(x=right_eye_x + ADJUSTMENTS[10]['x'], y=right_eye_y + ADJUSTMENTS[10]['y']))

        adjusted_nose_tip = adjust_point(points[7].position, ADJUSTMENTS[11])
        desired_landmarks.append(adjusted_nose_tip)

        desired_landmarks.append(adjust_point(points[8].position, ADJUSTMENTS[12]))
        desired_landmarks.append(adjust_point(points[9].position, ADJUSTMENTS[13]))
        desired_landmarks.append(adjust_point(points[10].position, ADJUSTMENTS[14]))
        desired_landmarks.append(adjust_point(points[11].position, ADJUSTMENTS[15]))

        for landmark in desired_landmarks:
            x = int(landmark.x)
            y = int(landmark.y)
            cv2.putText(gray_img_colored, 'x', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.1, (0, 0, 255), 1, cv2.LINE_AA)

    # Guardar la imagen procesada en una ruta absoluta
    output_image_path = 'static/output.png'
    output_img = cv2.resize(gray_img_colored, (400, 400))
    cv2.imwrite(output_image_path, output_img)
    os.remove(image_path)  # Elimina la imagen original

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload():
    file = request.files['file']
    if file:
        # Usar una ruta absoluta para la imagen guardada
        image_path = 'static/' + file.filename
        file.save(image_path)
        process_image(image_path)
        return render_template('output.html')

if __name__ == '__main__':
    app.run(debug=True)
