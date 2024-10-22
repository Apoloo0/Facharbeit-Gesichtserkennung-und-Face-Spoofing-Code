import cv2
import os
import imutils

# Pfad, unter dem alle Personendaten gespeichert werden
dataPath = 'dataset/' 

# Den Ordnernamen dynamisch als 'personX' zuweisen
existing_folders = [f for f in os.listdir(dataPath) if os.path.isdir(os.path.join(dataPath, f)) and f.startswith('person')]
next_person_id = len(existing_folders) + 1
personName = f'person{next_person_id}' 
# Erstellung den Pfad für die Daten der neuen Person
personPath = os.path.join(dataPath, personName)

# Oder Bilder eines bestehenden Pfades neu erstellen:
#personPath = 'dataPath/personX'

# Ordner erstellen, wenn er nicht existiert:
if not os.path.exists(personPath):
    print('Folder created: ', personPath)
    os.makedirs(personPath)

# Start die LIVE-KAMERA-Aufnahme:
#cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Oder Start mit der Videoaufzeichnung von einer 'personX':
cap = cv2.VideoCapture('other_videos/the_rock.mp4') 

# Gesichtsdetections-Klassifikator
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
count = 0

# Beginnen des Erfassens von Gesichtern:
while True:
    ret, frame = cap.read()
    if ret == False: 
        break
    
    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()

    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face = auxFrame[y:y+h, x:x+w]
        face = cv2.resize(face, (150, 150), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(os.path.join(personPath, 'image_{}.jpg'.format(count)), face)
        count += 1
    
    cv2.imshow('frame', frame)

    k = cv2.waitKey(1)
    if k == 27 or count >= 30:  # Beenden bei ESC oder nach X Bildern (gewünschte oder benötigte Anzahl von Bildern)
        break

cap.release()
cv2.destroyAllWindows()