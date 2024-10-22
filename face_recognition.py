import cv2
import os

dataPath = 'dataset/' # Pfad des Datensatzes
imagePaths = os.listdir(dataPath)
print('imagePaths=',imagePaths)

#  Erstellen eines Gesichtserkenners mit einer gewünschten Methode:
#face_recognizer = cv2.face.EigenFaceRecognizer_create()
#face_recognizer = cv2.face.FisherFaceRecognizer_create()
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Lesen des Modells nach der gewünschten Methode:
#face_recognizer.read('models/modelEigenFace.xml')
#face_recognizer.read('models/modelFisherFace.xml')
face_recognizer.read('models/modelLBPHFace.xml')

# Der Fokus liegt nur auf der LIVE-Kamera, da hier getestet werden soll, ob es Spoofing gibt oder nicht.
#cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap = cv2.VideoCapture('other_videos/the_rock.mp4')
#cap = cv2.VideoCapture('other_videos/test_videos/juan_cortez_test.mp4')

# Laden den Haar-Kaskaden-Klassifikator für die Gesichts- und Augendetektion:
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

# Endlosschleife zur kontinuierlichen Aufnahme von Bildern aus dem Live-Kamerabild
while True:
	ret,frame = cap.read()
	if ret == False: break
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Konvertieren das aufgenommene Bild in Graustufen, um die Gesichtserkennung zu vereinfachen.
	auxFrame = gray.copy() 

	# Gesichter im Graustufenbild mit dem Haar-Kaskaden-Klassifikator erkennen 
    # Parameter: 1.3 = Skalierungsfaktor, 5 = minimale Nachbarn für die Erkennung
	faces = faceClassif.detectMultiScale(gray,1.3,5) 

	# Wenn ein Gesicht erkannt wird, wird verarbeitet:
	for (x,y,w,h) in faces:
		face = auxFrame[y:y+h,x:x+w]
		face = cv2.resize(face,(150,150),interpolation= cv2.INTER_CUBIC)
		result = face_recognizer.predict(face) # Das Gesicht vorhersagen Methode

		cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)
		
		# Erkennung auf der Grundlage eines entsprechenden 'modelX.xml':
		'''
		# Mit EigenFaces
		if result[1] < 5100: # Schwellenwert für Eigenfaces-Methode
			cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
			cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
		else:
			cv2.putText(frame,'Unbekannt',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
			cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
		
		# Mit FisherFace
		if result[1] < 1500: # Schwellenwert für Fisherfaces-Methode
			cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
			cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
		else:
			cv2.putText(frame,'Unbekannt',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
			cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
		'''
  		# Mit LBPHFace
		if result[1] < 70: # Schwellenwert für LBPHFaces-Methode
			cv2.putText(frame,'{}'.format(imagePaths[result[0]]),(x,y-25),2,1.1,(0,255,0),1,cv2.LINE_AA)
			cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
		else:
			cv2.putText(frame,'Unbekannt',(x,y-20),2,0.8,(0,0,255),1,cv2.LINE_AA)
			cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,255),2)
    	
     
	cv2.imshow('frame',frame) # Anzeige des aktuellen Frames mit allen erkannten Flächen, Anmerkungen oder Überlagerungen in einem Fenster
	k = cv2.waitKey(1)
	if k == 27: # ESC zum Beenden 
		break

cap.release()
cv2.destroyAllWindows()