import cv2
import os

dataPath = 'dataset/'  # Pfad des Datensatzes
imagePaths = os.listdir(dataPath)
print('imagePaths =', imagePaths)

# Erstellen eines Gesichtserkenners mit einer gewünschten Methode:
#face_recognizer = cv2.face.EigenFaceRecognizer_create()
#face_recognizer = cv2.face.FisherFaceRecognizer_create()
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Lesen des Modells nach der gewünschten Methode:
#face_recognizer.read('models/modelEigenFace.xml')
#face_recognizer.read('models/modelFisherFace.xml')
face_recognizer.read('models/modelLBPHFace.xml')

# Der Fokus liegt nur auf der LIVE-Kamera, da hier getestet werden soll, ob es Spoofing gibt oder nicht.
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Laden den Haar-Kaskaden-Klassifikator für die Gesichts- und Augendetektion:
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eyeClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Variablen zur Blinzelerkennung
blink_count = 0
max_blinks = 3  # Anzahl der Blinzler zur Bestätigung eines echten Gesichts
blink_frame_count = 0  # Bildzähler für Blinkerkennung
current_face_id = None  # Verfolgung der aktuell erkannten Gesichts-ID
spoof_detected = False  # Verfolgen, ob ein Spoofing-Versuch entdeckt wird

frame_counter = 0
display_counter = False  # Flagge zur Kontrolle, ob der Zähler angezeigt werden soll

# Endlosschleife zur kontinuierlichen Aufnahme von Bildern aus dem Live-Kamerabild
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Inkrementieren des Bildzählers
    frame_counter += 1
    
    # Druckt die aktuelle Bildanzahl
    cv2.putText(frame, f'Frame Count: {frame_counter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Konvertieren das aufgenommene Bild in Graustufen, um die Gesichtserkennung zu vereinfachen.
    auxFrame = gray.copy() 
    
    # Gesichter im Graustufenbild mit dem Haar-Kaskaden-Klassifikator erkennen 
    # Parameter: 1.3 = Skalierungsfaktor, 5 = minimale Nachbarn für die Erkennung
    faces = faceClassif.detectMultiScale(gray, 1.3, 5) 

    if len(faces) > 0:
        # Wenn ein Gesicht erkannt wird, wird verarbeitet
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (150, 150), interpolation=cv2.INTER_CUBIC)
            result = face_recognizer.predict(face_resized) # Das Gesicht vorhersagen Methode

            # Wenn sich die aktuell erkannte Gesichts-ID ändert, setzen die Blinkzahl zurück.
            if current_face_id is None or current_face_id != result[0]:
                current_face_id = result[0]
                blink_count = 0  # Zurücksetzen des Blinzelzählers für ein neues Gesicht
                blink_frame_count = 0  # Rücksetzen des Blinkrahmenzählers
                spoof_detected = False  # Spoof-Erkennung zurücksetzen
                frame_counter = 0  # Bildzähler zurücksetzen
                display_counter = True  # Anzeigen des Zählers für das neue Gesicht 
                print(f"New face detected: {imagePaths[result[0]]}")

            cv2.putText(frame, '{}'.format(result), (x, y-5), 1, 1.3, (255, 255, 0), 1, cv2.LINE_AA)
            
            # Erkennung auf der Grundlage eines entsprechenden 'modelX.xml':
            '''
            # Mit Eigenfaces
            if result[1] < 5100: # Schwellenwert für Eigenfaces-Methode
                cv2.putText(frame, '{}'.format(imagePaths[result[0]]), (x, y-25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'Unbekannt', (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            # Mit Fisherfaces
            if result[1] < 1500: # Schwellenwert für Fisherfaces-Methode
                cv2.putText(frame, '{}'.format(imagePaths[result[0]]), (x, y-25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'Unbekannt', (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            '''
            # Mit LBPHFaces:
            if result[1] < 70: # Schwellenwert für LBPHFaces-Methode
                cv2.putText(frame, '{}'.format(imagePaths[result[0]]), (x, y-25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'Unbekannt', (x, y - 20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
            # Augen im Gesichtsbereich erkennen
            roi_gray = gray[y:y + h, x:x + w]
            eyes = eyeClassif.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))

            if len(eyes) == 2:
                # Die Anzahl der Blinzelbilder zurücksetzen, da Augen erkannt werden
                blink_frame_count = 0
                spoof_detected = False  # Zurücksetzen der Spoof-Erkennung, da wir Augen haben

                # Zeichnen (zur besseren Visualisierung) von Rechtecke um die erkannten Augen
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 0, 0), 2)


            else:
                # Die Augen fehlen, betrachte es als "Blinzeln"
                blink_count += 1
                cv2.putText(frame, 'Blink detected!', (x, y - 45), 2, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
                print(f"Blink count: {blink_count}")

            # Kombination von Blinzelzählung und möglicher Spoofing-Erkennung
            if blink_count >= max_blinks:
                cv2.putText(frame, 'Real Face Detected!', (x, y - 65), 2, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
                blink_frame_count = 0  # Rücksetzen des Blinkrahmenzählers bei Bestätigung
            elif frame_counter >= 70 and not spoof_detected:
                cv2.putText(frame, 'Possible Spoofing!', (x, y - 65), 2, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
                spoof_detected = True  # Setzen des Flages für die Spoof-Erkennung
                print("Spoofing detected!")
            else:
                # Wenn kein Blinken erkannt wird, wird die Anzahl der Bilder erhöht.
                blink_frame_count += 1  

            # Anzeige des Bildzählers, wenn ein neues Gesicht erkannt wird
            if display_counter:
                cv2.putText(frame, f'Frame Count: {frame_counter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                # Rücksetzen der Anzeige nach Erreichen des Zählerstandes von 100
                if frame_counter >= 100:
                    frame_counter = 1  # Rücksetzung auf 1 nach Erreichen von 100 Bilder

    else:
        # Wenn keine Gesichter erkannt werden, wird die aktuelle Gesichts-ID zurückgesetzt und der Bildzähler zurückgesetzt.
        current_face_id = None
        blink_frame_count += 1  # Erhöhen der Bildanzahl, wenn kein Gesicht erkannt wird

    cv2.imshow('frame', frame) # Anzeige des aktuellen Frames mit allen erkannten Flächen, Anmerkungen oder Überlagerungen in einem Fenster
    k = cv2.waitKey(1)
    if k == 27 or spoof_detected == True:  # ESC zum Beenden oder wenn Spoofing erkannt wird
        break

cap.release()
cv2.destroyAllWindows()