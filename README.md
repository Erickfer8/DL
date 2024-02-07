# Contador de personas en áreas

Este proyecto utiliza el modelo YOLO v8 (You Only Look Once) para la detección de personas, y realiza un seguimiento de las personas en áreas específicas.

<div align="center">
  <a href="https://postimg.cc/w3cvyL9r" target="_blank">
    <img src="https://i.postimg.cc/gc167HQG/Captura-de-pantalla-2024-02-06-200217.png" alt="Captura de Pantalla" width="400" height="300">
  </a>
</div>



 En que podemos aplicar este proyecto 

- Seguridad en Edificios y Espacios Públicos:

 - Implementar el sistema en entradas y salidas de edificios para contabilizar el número de personas que ingresan o salen.
 - Mejorar la seguridad mediante la identificación de áreas con concentración de personas.

- Transporte Público:

 - Aplicar el sistema en estaciones de tren, metro o autobús para gestionar el flujo de personas y mejorar la eficiencia en la gestión del transporte público.


Instalar las librerias necesarias 

Abre la terminal o consola de coomando y ejecuta los siguientes comandos.

`$pip install opencv-python`
`$pip install ultralytics `
`$pip install matplotlib `
`$pip install numpy `

Explicación del código 

El siguiente comando se utiliza para cargar el modelo  desde el archivo "models/yolov8n.pt" y configurar el modelo para la tarea de detección de objetos (task="detect").

    model = YOLO("models/yolov8n.pt", task="detect")


Funciones 
- La función draw_results dibuja cuadros delimitadores y etiquetas en la imagen para detecciones con confianza superior a 0.35.
- Opcionalmente, muestra el ID asociado a cada objeto detectado (show_id).

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


- La función get_center nos devuelve las coordenadas del centro de un cuadro delimitador para poder reaizar el conteo de personas en cada area.

		def get_center(bbox):
			center = ((bbox[0] + bbox[2] ) // 2, (bbox[1] + bbox[3]) //2)
			return center
			
- Este ciclo for  es esencial para la lógica de contar personas en áreas específicas del video basándose en el seguimiento de objetos proporcionado por el modelo YOLO.

		for box in results_track[0].boxes.xyxy:
				xc, yc = get_center(box.tolist())

				for i, area_coords in enumerate(areas, start=1):
					xi, yi, xf, yf = area(frame, * area_coords)
					if valid_detection(xc, yc, (xi, yi, xf, yf)):
						detections_por_area[i-1] += 1

				xc, yc = int(xc), int(yc)
				cv2.circle(img=frame, center=(xc, yc), radius=3, color=(0, 0, 255), thickness=-1)

- La función area devuelve las coordenadas ajustadas de un área rectangular en función de las dimensiones del frame.
  
		def area(frame, xi, yi, xf, yf):
			#info
			al, an, c = frame.shape
			#coordenadas 
			xi, yi = int(xi * an), int(yi * al)
			xf, yf = int(xf * an), int (yf * al)
			return xi , yi, xf, yf
			
- Con el siguiente ciclo for se dibuja las areas en el frame.

		for i, area_coords in enumerate(areas, start=1):
				xi, yi, xf, yf = area(frame, * area_coords)
				frame = draw_area(frame, color1, xi, yi, xf, yf)
				text = f'Area {i}'
				frame = draw_text(frame, color, text, xi + 4, yi + 25, size, tam, back = True)
