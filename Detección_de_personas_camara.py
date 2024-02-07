import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import random
from shapely.geometry import Polygon
import matplotlib.path as mplPath
import numpy as np


# coordenadas de la zona 
ZONE = np.array([
    [202, 90],
    [451, 98],
    [535, 225],
    [536, 302],
    [364, 376],
    [209, 293],
])


model = YOLO("models/yolov8n.pt", task="detect")

colors = random.choices(range(256), k=1000)

# obtener el centro de cada bbox 
def get_center(bbox):
    #xmin, ymin, xmax, ymax
    # 0     1     2     3
    center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
    return center


 
# dibujar el resultado de cada bbox
def draw_results(image, image_results, show_id=False):
    annotator = Annotator(image.copy())
    for box in image_results.boxes:
        b = box.xyxy[0]
        cls = int(box.cls)
        conf = float(box.conf)
        label = f"{model.names[cls]} {round(conf*100, 2)}"
        if show_id:
            label += f' id:{int(box.id)}'
        if cls == 0 and conf >= 0.35:
            annotator.box_label(b, label, color=colors[int(box.id):int(box.id)+2])
    image_annotated = annotator.result()
    return image_annotated


## funcion para validar deteccion en la zona
def validar_det(xc, yc):
    return mplPath.Path(ZONE).contains_point((xc, yc))


def detector(cap: object):

    
    while cap.isOpened():
        status, frame = cap.read()
        if not status:
            break
        
        
        results_track = model.track(frame, conf=0.40, classes=0, tracker="botsort.yaml", persist=True, verbose=False)

        preds = model(frame)

        detections = 0
        for box in results_track[0].boxes.xyxy:
            xc, yc = get_center(box)
            
            if validar_det(xc, yc):
                detections += 1
            
            cv2.circle(img=frame, center=(int(xc), int(yc)), radius=5, color=(0, 0, 255), thickness=-1)
            cv2.rectangle(img=frame, pt1=(int(box[0]), int(box[1])), pt2=(int(box[2]), int(box[3])), color=(0, 0, ), thickness=2)

        cv2.putText(img=frame, text=f"Cantidad de Personas: {detections}", org=(10,25), fontFace=cv2.FONT_ITALIC, fontScale=1, color=(0,0,255), thickness=1)
        cv2.polylines(img=frame, pts=[ZONE], isClosed=True, color=(255,0,0), thickness=2)

        image = draw_results(frame, results_track[0], show_id=True)

        cv2.imshow("Video con Detecciones", image)


        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
    cap.release()


if __name__ == '__main__':
    video = 0
    cap = cv2.VideoCapture(video)
    
    detector(cap)