# The following code detects and matches a pre-loaded face with a live feed video
import threading
import cv2 
from deepface import DeepFace

cap = cv2.VideoCapture (0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

counter = 0
#fps = 0

face_match = False

reference_img = cv2.imread("reference.jpg")


def check_face(frame):
  global face_match

  try:
    if DeepFace.verify (frame, reference_img.copy())['verified']:
      face_match=True
    else:
      face_match=False
  except ValueError: 
    face_match=False


while True:
 
 #if fps % 10 == 0:
   
  ret, frame = cap.read()

  if ret:
    if counter % 100 == 0:

      try:

        threading.Thread(target=check_face, args=(frame.copy(),)).start() 
      except ValueError:
        pass

    counter += 1

    if face_match:
      cv2.putText(frame, "MATCH!", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    else:
      cv2.putText(frame, "NO MATCH!", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    cv2.imshow("video", frame)

  #fps += 1
  key = cv2.waitKey(1)
  if key == ord("q"):
      break
  
cv2.destroyAllWindows()


# The bottom code is matching 2 faces from static images
'''
import cv2 
from deepface import DeepFace

# Load the reference images
reference_img1 = cv2.imread("reference.jpg")
reference_img2 = cv2.imread("reference4.jpg")

# Compare the images
try:
    result = DeepFace.verify(reference_img1, reference_img2)
    if result['verified']:
        print("\nMATCH!")
    else:
        print("\nNO MATCH!")
except ValueError: 
    print("\nError in comparison!")
'''