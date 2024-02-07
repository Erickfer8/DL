import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import random
from shapely.geometry import Polygon
import matplotlib.path as mplPath
import numpy as np


videopath = "video.webm"


model = YOLO("models/yolov8m.pt", task="detect")

colors = random.choices(range(256), k=1000)

#Función  de los resultados de cada bbox
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

video = cv2.VideoCapture(videopath)


### función para las areas 
def area(frame, xi, yi, xf, yf):
    #info
    al, an, c = frame.shape
    #coordenadas 
    xi, yi = int(xi * an), int(yi * al)
    xf, yf = int(xf * an), int (yf * al)
    return xi , yi, xf, yf


# función para difijar las areas 
def  draw_area (frame, color, xi, yi, xf, yf ): 
     img = cv2.rectangle ( frame, (xi,yi), (xf,yf), color, 2, 2)
     return img




# función para el texto 

def draw_text(img, color,text,xi, yi, size, tam, back = False):
    sizetext = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, size, tam)
    dim = sizetext[0]
    baseline = sizetext[1]
    if back == True:
        img = cv2.rectangle(img,  (xi, yi - dim[1] - baseline), (xi +  dim[0], yi + baseline - 7), (0,0,0), cv2.FILLED)
    
    
    img = cv2.putText(img, text, (xi, yi - 5), cv2.FONT_ITALIC, size, color, tam )

    return img

## función para obtener el centro de cada bbox

def get_center(bbox):
    center = ((bbox[0] + bbox[2] ) // 2, (bbox[1] + bbox[3]) //2)
    return center



##  dtenerminar cantidad de personas en el area 
def valid_detection(xc, yc, area_coords):
    area_detectada = mplPath.Path([(area_coords[0], area_coords[1]), (area_coords[2], area_coords[1]),
                                (area_coords[2], area_coords[3]), (area_coords[0], area_coords[3])])
    return area_detectada.contains_point((xc, yc))


## DEFINIMOS LAS AREAS 
areas = [
    (0.0151, 0.0186, 0.5539, 0.5444),   # A1
    (0.5739, 0.0186, 0.9649, 0.4050),   # A2
    (0.5739, 0.4250, 0.9649, 0.9444),   # A3
    (0.0151, 0.5550, 0.5539, 0.9444)    # A4
]


while True:
    ret, frame = video.read()
    if not ret:
        break
    

    # Dibujar áreas
    color1 = (255, 0, 0)
    color = (255, 255, 0)
    size, tam = 0.65, 1

    for i, area_coords in enumerate(areas, start=1):
        xi, yi, xf, yf = area(frame, * area_coords)
        frame = draw_area(frame, color1, xi, yi, xf, yf)
        text = f'Area {i}'
        frame = draw_text(frame, color, text, xi + 4, yi + 25, size, tam, back = True)
     

    results_track = model.track(frame, conf=0.40, classes=0, tracker="botsort.yaml", persist=True, verbose=False)
    
    # Contar personas en cada área
    detections_por_area = [0] * len(areas)

    for box in results_track[0].boxes.xyxy:
        xc, yc = get_center(box.tolist())

        for i, area_coords in enumerate(areas, start=1):
            xi, yi, xf, yf = area(frame, * area_coords)
            if valid_detection(xc, yc, (xi, yi, xf, yf)):
                detections_por_area[i-1] += 1

        xc, yc = int(xc), int(yc)
        cv2.circle(img=frame, center=(xc, yc), radius=3, color=(0, 0, 255), thickness=-1)
    

    

    
    # Mostrar la cantidad de personas por área
    # Coordenadas específicas para colocar el texto en cada área
    text_positions = {
        'A1': (230, 36),
        'A2': (550, 36),
        'A3': (445, 539),
        'A4': (15, 539)
    }

    # Iterar sobre las áreas y mostrar la cantidad de personas en cada una
    for i, count in enumerate(detections_por_area, start=1):
        area_key = f'A{i}'
        
        # Obtener las coordenadas del área
        xi, yi, _, _ = area(frame, *areas[i-1])
        
        # Obtener las coordenadas específicas para colocar el texto
        text_x, text_y = text_positions.get(area_key, (xi + 10, yi + 30))
        
        text = f'{area_key}: {count} personas'
        frame = draw_text(frame, color, text, text_x, text_y, size, tam, back = True)




    image = draw_results(frame, results_track[0], show_id=True)

    cv2.imshow("Video con Detecciones", image)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

