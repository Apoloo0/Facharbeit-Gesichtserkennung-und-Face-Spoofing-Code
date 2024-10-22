import cv2
import os
import numpy as np

dataPath = 'dataset/' # Pfad des Datensatzes
peopleList = os.listdir(dataPath)
print('List of people: ', peopleList)

# Initialisieren leerer Listen zum Speichern von Beschriftungen und Gesichtsbilddaten
labels = []
facesData = []
label = 0  # Diese Variable ordnet jeder Person eine eindeutige numerische Bezeichnung zu (Etikett)

# Schleife durch die Ordner der einzelnen Personen im Datensatz
for nameDir in peopleList:
    personPath = dataPath + '/' + nameDir
    print('Reading the images')

    # Schleife durch jede Bilddatei für die aktuelle Person
    for fileName in os.listdir(personPath):
        print('Faces: ', nameDir + '/' + fileName)

        # Anhängen des aktuellen Etiketts (das die Person darstellt) an die Etikettenliste
        labels.append(label)

        # Lesen des Bild in Graustufen und Anfügen an die Liste facesData
        facesData.append(cv2.imread(personPath + '/' + fileName, 0))

    # Erhöhen des Etikett nach der Verarbeitung aller Bilder für eine Person
    label += 1

# Methoden zum Trainieren des Erkenners nach der gewünschten Methode:
face_recognizer = cv2.face.EigenFaceRecognizer_create()
#face_recognizer = cv2.face.FisherFaceRecognizer_create()
#face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Training des Erkenners
print("Training...")
face_recognizer.train(facesData, np.array(labels))

# Speichern des nach der gewünschten Methode erstellten Modells:
face_recognizer.write('models/modelEigenFace.xml')
#face_recognizer.write('models/modelFisherFace.xml')
#face_recognizer.write('models/modelLBPHFace.xml')

print('the model is stored!')
