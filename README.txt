Aus Gründen der Privatsphäre der Personen, mit denen der Code geübt wurde, 
wurden nur Videos und Bilder von Personen des öffentlichen Lebens belassen. 

Der Code „face_capture“ ist dafür zuständig, im Ordner „datasets/“ 
die gewünschte Anzahl von Bildern für eine „personX“ anhand eines Videos oder der Webcam zu erstellen.

Der Code „training_facial_recognition“ verwendet die wichtigsten Gesichtserkennungsmethoden 
(Eigenfaces, Fisherfaces, LBPH), um ein xml-Modell anhand der Bilder in „datasets/“ zu trainieren, 
und das Modell wird in „models/“ gespeichert.

Der Code „face_recognition“ ist dafür zuständig, 
eine der Gesichtserkennungsmethoden + Ihr trainiertes Modell zu verwenden, 
um das Gesicht vor Ihnen zu erkennen 
(es ist wichtig, die anderen Methoden als Kommentare zu hinterlassen).

Die obigen Codes sind die Originale von Gabriela Solano mit leichten 
Änderungen von mir, aber die Credits für die ersten 3 sind ihre: 
https://github.com/GabySol/OmesTutorials2020/tree/master/6%20RECONOCIMIENTO%20FACIAL

Der Code „face_recognition_with_blink_detection“ wiederum funktioniert 
genauso wie „face_recognition“, erkennt aber anhand des Blinzelns, 
ob es sich bei dem Gesicht vor Ihnen um einen Versuch des Face Spoofing handelt oder nicht.
