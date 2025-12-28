# # import cv2
# # import numpy as np
# # import os
# # from ultralytics import YOLO
# # import onnxruntime as ort
# # from scipy.spatial.distance import cosine
# #
# # # === Load models ===
# # face_detector = YOLO("yolov8n-face-lindevs.pt")  # <-- your YOLOv8n .pt file
# # arcface = ort.InferenceSession("arcface.onnx")
# #
# # # === Load player profiles ===
# # profile_dir = "static/profile_pics"
# # profiles = {}
# # for filename in os.listdir(profile_dir):
# #     if filename.lower().endswith((".jpg", ".png")):
# #         img = cv2.imread(os.path.join(profile_dir, filename))
# #         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# #         img = cv2.resize(img, (112, 112))
# #         img = np.transpose(img, (2, 0, 1)).astype(np.float32)
# #         img = (img / 255.0 - 0.5) / 0.5
# #         embedding = arcface.run(None, {"data": np.expand_dims(img, 0)})[0][0]
# #         embedding = embedding / np.linalg.norm(embedding)
# #         profiles[filename] = embedding
# #
# # print(f"Loaded player profiles: {list(profiles.keys())}")
# #
# # # === Helper ===
# # def detect_faces(frame):
# #     results = face_detector.predict(frame, imgsz=640, conf=0.5, verbose=False)
# #     boxes = []
# #     h, w = frame.shape[:2]
# #     for r in results:
# #         for box in r.boxes.xyxy.cpu().numpy():
# #             x1, y1, x2, y2 = box
# #             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
# #             boxes.append([x1, y1, x2, y2])
# #     return boxes
# #
# # def get_face_embedding(face_img):
# #     if face_img is None or face_img.size == 0:
# #         return None
# #     face = cv2.resize(face_img, (112, 112))
# #     face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
# #     face = np.transpose(face, (2, 0, 1)).astype(np.float32)
# #     face = (face / 255.0 - 0.5) / 0.5
# #     emb = arcface.run(None, {"data": np.expand_dims(face, 0)})[0][0]
# #     emb = emb / np.linalg.norm(emb)
# #     return emb
# #
# # def match_face(embedding):
# #     if embedding is None:
# #         return "Unknown"
# #     best_match = "Unknown"
# #     best_score = 1.0
# #     for name, ref_emb in profiles.items():
# #         dist = cosine(embedding, ref_emb)
# #         if dist < best_score:
# #             best_score = dist
# #             best_match = name
# #     return best_match if best_score < 0.5 else "Unknown"
# #
# # # === Process video ===
# # cap = cv2.VideoCapture("match1.mp4")
# # fps = cap.get(cv2.CAP_PROP_FPS)
# # print(f"Video FPS: {fps}")
# #
# # frame_skip = 2  # every 2nd frame
# # frame_count = 0
# # boxes_names = []
# #
# # while True:
# #     ret, frame = cap.read()
# #     if not ret:
# #         break
# #
# #     if frame_count % frame_skip == 0:
# #         boxes = detect_faces(frame)
# #         boxes_names = []
# #         for box in boxes:
# #             x1, y1, x2, y2 = box
# #             face_crop = frame[y1:y2, x1:x2]
# #             embedding = get_face_embedding(face_crop)
# #             name = match_face(embedding)
# #             if name != "Unknown":
# #                 boxes_names.append((box, name))
# #
# #     # Draw matched only
# #     for box, name in boxes_names:
# #         x1, y1, x2, y2 = box
# #         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
# #         cv2.putText(frame, name, (x1, y1 - 10),
# #                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
# #
# #     cv2.imshow("Recognize Players", frame)
# #     if cv2.waitKey(int(1000 / fps)) == ord("q"):
# #         break
# #
# #     frame_count += 1
# #
# # cap.release()
# # cv2.destroyAllWindows()
#
#
# import cv2
# import numpy as np
# import face_recognition
# import os
#
# # === Setup ===
#
# # Folder with known player images
# path = 'static/profile_pics'
# images = []
# classNames = []
# myList = os.listdir(path)
# print("Found profile pictures:", myList)
#
# for cl in myList:
#     curImg = cv2.imread(f'{path}/{cl}')
#     images.append(curImg)
#     classNames.append(os.path.splitext(cl)[0])
# print("Loaded player names:", classNames)
#
# # === Encode known faces ===
#
# def findEncodings(images):
#     encodeList = []
#     for img in images:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         encodes = face_recognition.face_encodings(img)
#         if len(encodes) > 0:
#             encodeList.append(encodes[0])
#         else:
#             print("Warning: No face found in", img)
#     return encodeList
#
# encodeListKnown = findEncodings(images)
# print('Encoding Complete. Total known faces:', len(encodeListKnown))
#
# # === Process video instead of webcam ===
#
# video_path = "match1.mp4"
# cap = cv2.VideoCapture(video_path)
#
# frame_skip = 0  # optional: skip every 2nd frame for speed
# frame_count = 0
#
# while True:
#     success, img = cap.read()
#     if not success:
#         break
#
#     # Resize for faster processing
#     imgS = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
#     imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
#
#     # Detect faces + get encodings
#     facesCurFrame = face_recognition.face_locations(imgS)
#     encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
#
#     # Compare with known players
#     for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
#         matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
#         faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
#         matchIndex = np.argmin(faceDis)
#
#         if matches[matchIndex]:
#             name = classNames[matchIndex].upper()
#             print(f"Detected: {name}")
#
#             # Scale back up to original frame size
#             y1, x2, y2, x1 = faceLoc
#             y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
#
#             cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
#             cv2.putText(img, name, (x1+6, y2-6),
#                         cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
#
#     # Show frame
#     cv2.imshow('Video Face Recognition', img)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
#     frame_count += 1
#
# cap.release()
# cv2.destroyAllWindows()
# # import cv2
# # import insightface
# # import numpy as np
# # import os
# #
# # # ================
# # # CONFIGURATION
# # # ================
# #
# # # 1️⃣ Path to known player face images
# # KNOWN_FACE_DIR = "static/profile_pics"
# #
# # # 2️⃣ Path to input video file (or 0 for webcam)
# # VIDEO_PATH = "match1.mp4"   # <-- replace with your video file
# #
# # # 3️⃣ Matching threshold (lower is stricter)
# # MATCH_THRESHOLD = 1.2
# #
# # # ================
# # # Load InsightFace
# # # ================
# #
# # print("[INFO] Loading InsightFace (RetinaFace + ArcFace)...")
# # model = insightface.app.FaceAnalysis(name='buffalo_l')
# # model.prepare(ctx_id=0, det_size=(640, 640))
# #
# # # ================================
# # # Encode known player face images
# # # ================================
# #
# # print("[INFO] Encoding known players...")
# # known_embeddings = []
# # known_names = []
# #
# # if not os.path.exists(KNOWN_FACE_DIR):
# #     print(f"[ERROR] Directory '{KNOWN_FACE_DIR}' does not exist.")
# #     exit()
# #
# # for filename in os.listdir(KNOWN_FACE_DIR):
# #     if filename.lower().endswith((".jpg", ".png", ".jpeg")):
# #         name = os.path.splitext(filename)[0]
# #         img_path = os.path.join(KNOWN_FACE_DIR, filename)
# #         img = cv2.imread(img_path)
# #
# #         if img is None:
# #             print(f"[WARN] Cannot read {img_path}")
# #             continue
# #
# #         faces = model.get(img)
# #         if faces:
# #             emb = faces[0].embedding
# #             known_embeddings.append(emb)
# #             known_names.append(name)
# #             print(f"[INFO] Added: {name}")
# #         else:
# #             print(f"[WARN] No face detected in: {filename}")
# #
# # print(f"[INFO] Total known players: {len(known_names)}")
# #
# # # ================================
# # # Process video file or webcam
# # # ================================
# #
# # cap = cv2.VideoCapture(VIDEO_PATH)
# #
# # if not cap.isOpened():
# #     print(f"[ERROR] Cannot open video source: {VIDEO_PATH}")
# #     exit()
# #
# # print("[INFO] Starting player recognition on video...")
# #
# # while True:
# #     ret, frame = cap.read()
# #     if not ret:
# #         print("[INFO] Video ended or cannot read frame.")
# #         break
# #
# #     faces = model.get(frame)
# #
# #     for face in faces:
# #         bbox = face.bbox.astype(int)
# #         emb = face.embedding
# #
# #         # Compare with known embeddings
# #         name = "Unknown"
# #         min_dist = float('inf')
# #
# #         for db_emb, db_name in zip(known_embeddings, known_names):
# #             dist = np.linalg.norm(emb - db_emb)
# #             if dist < min_dist:
# #                 min_dist = dist
# #                 name = db_name
# #
# #         if min_dist > MATCH_THRESHOLD:
# #             name = "Unknown"
# #
# #         # Draw bounding box & name
# #         cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
# #                       (0, 255, 0), 2)
# #         cv2.putText(frame, f"{name} ({min_dist:.2f})",
# #                     (bbox[0], bbox[1] - 10),
# #                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
# #
# #     cv2.imshow("Football Player Face Recognition", frame)
# #
# #     if cv2.waitKey(1) & 0xFF == ord('q'):
# #         print("[INFO] Quitting...")
# #         break
# #
# # cap.release()
# # cv2.destroyAllWindows()
# import cv2
# import insightface
# import numpy as np
# import os
#
# # ==============================
# # CONFIGURATION
# # ==============================
#
# KNOWN_FACE_DIR = "static/profile_pics"  # Folder for player images
# VIDEO_PATH = "match1.mp4"               # Your match video
# MATCH_THRESHOLD = 25.0  # realistic for your embedding scale
# # Slightly relaxed for real match conditions
#
# # ==============================
# # Initialize InsightFace
# # ==============================
#
# print("[INFO] Loading InsightFace model (buffalo_l)...")
# model = insightface.app.FaceAnalysis(name='buffalo_l')
# model.prepare(ctx_id=0, det_size=(640, 640))
#
# # ==============================
# # Load known player faces & encode
# # ==============================
#
# print("[INFO] Encoding known player faces...")
# known_embeddings = []
# known_names = []
#
# for filename in os.listdir(KNOWN_FACE_DIR):
#     if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
#         name = os.path.splitext(filename)[0]
#         img = cv2.imread(os.path.join(KNOWN_FACE_DIR, filename))
#         if img is None:
#             print(f"[WARN] Cannot read: {filename}")
#             continue
#         faces = model.get(img)
#         if faces:
#             known_embeddings.append(faces[0].embedding)
#             known_names.append(name)
#             print(f"[OK] Added: {name}")
#         else:
#             print(f"[WARN] No face found in: {filename}")
#
# print(f"[INFO] Total known faces: {len(known_names)}")
#
# # ==============================
# # Video processing + tracking
# # ==============================
#
# cap = cv2.VideoCapture(VIDEO_PATH)
# if not cap.isOpened():
#     print(f"[ERROR] Cannot open video: {VIDEO_PATH}")
#     exit()
#
# # === Simple tracker ===
# trackers = []
# track_id = 0
# MAX_LOST = 10  # frames to keep lost face
#
# def iou(b1, b2):
#     """Intersection over Union"""
#     x1 = max(b1[0], b2[0])
#     y1 = max(b1[1], b2[1])
#     x2 = min(b1[2], b2[2])
#     y2 = min(b1[3], b2[3])
#     inter = max(0, x2 - x1) * max(0, y2 - y1)
#     area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
#     area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
#     union = area1 + area2 - inter
#     return inter / union if union > 0 else 0
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     faces = model.get(frame)
#
#     # detections = []
#     # for face in faces:
#     #     emb = face.embedding
#     #     bbox = face.bbox.astype(int)
#     #     name = "Unknown"
#     #     min_dist = float('inf')
#     #
#     #     for db_emb, db_name in zip(known_embeddings, known_names):
#     #         dist = np.linalg.norm(emb - db_emb)
#     #         if dist < min_dist:
#     #             min_dist = dist
#     #             name = db_name
#     #
#     #     if min_dist > MATCH_THRESHOLD:
#     #         name = "Unknown"
#     #
#     #     print(f"[DEBUG] Match: {name} | Distance: {min_dist:.3f} | Threshold: {MATCH_THRESHOLD}")
#     detections = []
#     for face in faces:
#         emb = face.embedding
#         bbox = face.bbox.astype(int)
#         name = None
#         min_dist = float('inf')
#
#         for db_emb, db_name in zip(known_embeddings, known_names):
#             dist = np.linalg.norm(emb - db_emb)
#             if dist < min_dist:
#                 min_dist = dist
#                 name = db_name
#
#         if min_dist <= MATCH_THRESHOLD:
#             print(f"[DEBUG] Match: {name} | Distance: {min_dist:.3f}")
#             detections.append({'bbox': bbox, 'embedding': emb, 'name': name})
#         else:
#             print(f"[DEBUG] Unknown face ignored | Distance: {min_dist:.3f}")
#
#         detections.append({'bbox': bbox, 'embedding': emb, 'name': name})
#
#     # === Update trackers ===
#     for det in detections:
#         matched = False
#         for trk in trackers:
#             if iou(trk['bbox'], det['bbox']) > 0.3:
#                 trk['bbox'] = det['bbox']
#                 trk['embedding'] = det['embedding']
#                 trk['name'] = det['name']
#                 trk['lost'] = 0
#                 matched = True
#                 break
#         if not matched:
#             trackers.append({
#                 'id': track_id,
#                 'bbox': det['bbox'],
#                 'embedding': det['embedding'],
#                 'name': det['name'],
#                 'lost': 0
#             })
#             track_id += 1
#
#     # === Increment lost counters & remove old trackers ===
#     for trk in trackers:
#         trk['lost'] += 1
#     trackers = [trk for trk in trackers if trk['lost'] <= MAX_LOST]
#
#     # === Draw ===
#     for trk in trackers:
#         bbox = trk['bbox']
#         name = trk['name']
#         cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),(0, 255, 0), 2)
#         cv2.putText(frame, f"{name}", (bbox[0], bbox[1] - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#
#     cv2.imshow("Player Face Tracking", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
#############upr wala sai hai
# import cv2
# import numpy as np
# import os
# from ultralytics import YOLO
# import insightface
#
# # ===========================
# # CONFIG
# # ===========================
#
# PLAYER_MODEL_PATH = "yolov8n.pt"  # COCO model
# KNOWN_FACE_DIR = "static/profile_pics"
# VIDEO_PATH = "match1.mp4"
# MATCH_THRESHOLD = 25.0
#
# # ===========================
# # Load YOLOv8n player detector
# # ===========================
#
# print("[INFO] Loading YOLOv8n for player detection...")
# player_detector = YOLO(PLAYER_MODEL_PATH)
#
# # ===========================
# # Load InsightFace for face detection + embedding
# # ===========================
#
# print("[INFO] Loading InsightFace ArcFace model...")
# face_model = insightface.app.FaceAnalysis(name='buffalo_l')
# face_model.prepare(ctx_id=0, det_size=(224, 224))
#
# # ===========================
# # Encode known player faces
# # ===========================
#
# print("[INFO] Encoding known player faces...")
# known_embeddings = []
# known_names = []
#
# for filename in os.listdir(KNOWN_FACE_DIR):
#     if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
#         name = os.path.splitext(filename)[0]
#         img = cv2.imread(os.path.join(KNOWN_FACE_DIR, filename))
#         if img is None:
#             print(f"[WARN] Cannot read: {filename}")
#             continue
#         faces = face_model.get(img)
#         if faces:
#             known_embeddings.append(faces[0].embedding)
#             known_names.append(name)
#             print(f"[OK] Added: {name}")
#         else:
#             print(f"[WARN] No face found in: {filename}")
#
# print(f"[INFO] Total known faces: {len(known_names)}")
#
# # ===========================
# # Process video
# # ===========================
#
# cap = cv2.VideoCapture(VIDEO_PATH)
# if not cap.isOpened():
#     print(f"[ERROR] Cannot open video: {VIDEO_PATH}")
#     exit()
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # === 1) Detect PERSON only ===
#     results = player_detector(frame)
#     player_boxes = []
#     for r in results:
#         for box, cls, conf in zip(r.boxes.xyxy.cpu().numpy(),
#                                   r.boxes.cls.cpu().numpy(),
#                                   r.boxes.conf.cpu().numpy()):
#             x1, y1, x2, y2 = box.astype(int)
#
#             # Filter: COCO class 0 == person
#             if int(cls) == 0 and conf > 0.5:
#                 w, h = x2 - x1, y2 - y1
#                 # Optional: filter by size, to reduce audience detection
#                 if h > 100 and w < h:
#                     player_boxes.append((x1, y1, x2, y2))
#
#     # === 2) Inside each player box, detect face + recognize ===
#     for (x1, y1, x2, y2) in player_boxes:
#         player_crop = frame[y1:y2, x1:x2]
#         if player_crop.size == 0:
#             continue
#
#         faces = face_model.get(player_crop)
#         for face in faces:
#             fbox = face.bbox.astype(int)
#             fx1, fy1, fx2, fy2 = fbox
#
#             # Map face box back to original frame coords
#             fx1 += x1
#             fy1 += y1
#             fx2 += x1
#             fy2 += y1
#
#             emb = face.embedding
#             name = " "
#             min_dist = float('inf')
#             for db_emb, db_name in zip(known_embeddings, known_names):
#                 dist = np.linalg.norm(emb - db_emb)
#                 if dist < min_dist:
#                     min_dist = dist
#                     name = db_name
#
#             if min_dist > MATCH_THRESHOLD:
#                 name = " "
#
#             cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
#             cv2.putText(frame, f"{name} ({min_dist:.2f})", (fx1, fy1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#
#         # Draw player box for reference
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
#
#     cv2.imshow("Player + Face Recognition", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()


import cv2
import numpy as np
import os
from ultralytics import YOLO
import insightface

# ===========================
# CONFIG
# ===========================

PLAYER_MODEL_PATH = "yolov8n.pt"  # your COCO YOLO model
KNOWN_FACE_DIR = "tantri"
VIDEO_PATH = r'D:\dataset\goals\goals (4).mp4'
MATCH_THRESHOLD = 25.0

# ===========================
# Load YOLOv8n player detector
# ===========================

print("[INFO] Loading YOLOv8n for player detection...")
player_detector = YOLO(PLAYER_MODEL_PATH)

# ===========================
# Load InsightFace for face detection + embedding
# ===========================

print("[INFO] Loading InsightFace ArcFace model...")
face_model = insightface.app.FaceAnalysis(name='buffalo_l')
face_model.prepare(ctx_id=0, det_size=(224, 224))

# ===========================
# Encode known player faces
# ===========================

print("[INFO] Encoding known player faces...")
known_embeddings = []
known_names = []

for filename in os.listdir(KNOWN_FACE_DIR):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        name = os.path.splitext(filename)[0]
        img = cv2.imread(os.path.join(KNOWN_FACE_DIR, filename))
        if img is None:
            print(f"[WARN] Cannot read: {filename}")
            continue
        faces = face_model.get(img)
        if faces:
            known_embeddings.append(faces[0].embedding)
            known_names.append(name)
            print(f"[OK] Added: {name}")
        else:
            print(f"[WARN] No face found in: {filename}")

print(f"[INFO] Total known faces: {len(known_names)}")

# ===========================
# Process video
# ===========================

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"[ERROR] Cannot open video: {VIDEO_PATH}")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # === 1) Detect PERSON only ===
    results = player_detector(frame)
    player_boxes = []
    for r in results:
        for box, cls, conf in zip(r.boxes.xyxy.cpu().numpy(),
                                  r.boxes.cls.cpu().numpy(),
                                  r.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = box.astype(int)

            # Filter: COCO class 0 == person
            if int(cls) == 0 and conf > 0.5:
                w, h = x2 - x1, y2 - y1
                # Optional: skip tiny or weird shapes (crowd)
                if h > 100 and w < h:
                    player_boxes.append((x1, y1, x2, y2))

    # === 2) Inside each player box, detect face + recognize ===
    for (x1, y1, x2, y2) in player_boxes:
        player_crop = frame[y1:y2, x1:x2]
        if player_crop.size == 0:
            continue

        faces = face_model.get(player_crop)
        for face in faces:
            fbox = face.bbox.astype(int)
            fx1, fy1, fx2, fy2 = fbox

            # Map face box back to original frame coords
            fx1 += x1
            fy1 += y1
            fx2 += x1
            fy2 += y1

            emb = face.embedding
            name = None
            min_dist = float('inf')
            for db_emb, db_name in zip(known_embeddings, known_names):
                dist = np.linalg.norm(emb - db_emb)
                if dist < min_dist:
                    min_dist = dist
                    name = db_name

            # Only draw if match is within threshold
            if min_dist <= MATCH_THRESHOLD:
                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} ({min_dist:.2f})", (fx1, fy1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Draw player box for reference (optional)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imshow("Player + Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
