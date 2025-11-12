import cv2
import mediapipe as mp
import numpy as np
import time
from PIL import ImageFont, ImageDraw, Image
from collections import deque, Counter

# ----------------- Initialize Mediapipe -----------------
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2)

cap = cv2.VideoCapture(0)

# ----------------- Floating Emoji Parameters -----------------
floating_emoji = None
floating_emoji_x = None
floating_emoji_y = None
floating_text = None
floating_start_time = None

detection_history = deque(maxlen=15)  # Keep last 15 frames for smoothing

# Load emoji-supporting font
emoji_font_path = "C:/Windows/Fonts/seguiemj.ttf"  # Segoe UI Emoji
emoji_font = ImageFont.truetype(emoji_font_path, 100)
text_font = ImageFont.truetype(emoji_font_path, 50)

# Gesture to emoji mapping
gesture_emoji_map = {
    "Heart": "❤️",
    "Facepalm": "🤦",
    "Thumbs Up": "👍",
    "Victory Hand": "✌️",
    "OK Hand": "👌",
    "Smile": "😄",
    "Peace Love": "✌️❤️",
    "Party": "🥳🎉",
    "High Five": "🙌",
    "Rock On": "🤘",
    "Clap": "👏"
}

# ----------------- Helper Functions -----------------
def set_floating_emoji(text, emoji, x, y):
    global floating_emoji, floating_text, floating_emoji_x, floating_emoji_y, floating_start_time
    floating_emoji = emoji
    floating_text = text
    floating_emoji_x = x
    floating_emoji_y = y
    floating_start_time = time.time()

def draw_pil_text_on_cv2(image, text, emoji, position, text_font, emoji_font):
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    text_position = (position[0] - 100, position[1] - 100)
    emoji_position = (position[0] - 50, position[1] - 50)
    draw.text(text_position, text, font=text_font, fill=(255, 255, 255, 255))
    draw.text(emoji_position, emoji, font=emoji_font, fill=(255, 255, 255, 255))
    return np.array(pil_image)

# ----------------- Gesture Detection Functions -----------------
def detect_thumbs_up(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    return thumb_tip.y < wrist.y and index_tip.y > thumb_tip.y

def detect_heart(left_hand, right_hand):
    left_thumb = left_hand.landmark[mp_hands.HandLandmark.THUMB_TIP]
    right_thumb = right_hand.landmark[mp_hands.HandLandmark.THUMB_TIP]
    left_index = left_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    right_index = right_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    thumb_dist = np.linalg.norm([left_thumb.x - right_thumb.x, left_thumb.y - right_thumb.y])
    index_dist = np.linalg.norm([left_index.x - right_index.x, left_index.y - right_index.y])
    return thumb_dist < 0.05 and index_dist < 0.05

def detect_ok_hand(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    return np.linalg.norm([thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y]) < 0.05

def detect_victory_hand(hand_landmarks):
    idx_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    mid_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    return idx_tip.y < ring_tip.y and mid_tip.y < pinky_tip.y

def detect_high_five(left_hand, right_hand):
    fingers_left = [left_hand.landmark[i].y < left_hand.landmark[mp_hands.HandLandmark.WRIST].y for i in range(0,21,4)]
    fingers_right = [right_hand.landmark[i].y < right_hand.landmark[mp_hands.HandLandmark.WRIST].y for i in range(0,21,4)]
    return all(fingers_left) and all(fingers_right)

def detect_facepalm(hand, face):
    forehead_y = (face.landmark[9].y + face.landmark[10].y)/2
    forehead_x = (face.landmark[9].x + face.landmark[10].x)/2
    fingers_y = [hand.landmark[i].y for i in [8,12,16,20]]
    fingers_x = [hand.landmark[i].x for i in [8,12,16,20]]
    return all(fy < forehead_y + 0.05 and abs(fx - forehead_x) < 0.1 for fy, fx in zip(fingers_y, fingers_x))

def detect_smile(face):
    left_mouth = face.landmark[61]
    right_mouth = face.landmark[291]
    upper_lip = face.landmark[13]
    lower_lip = face.landmark[14]
    width = np.linalg.norm([left_mouth.x - right_mouth.x, left_mouth.y - right_mouth.y])
    height = np.linalg.norm([upper_lip.x - lower_lip.x, upper_lip.y - lower_lip.y])
    ratio = width / height if height !=0 else 0
    return ratio > 2.0

# ----------------- Main Loop -----------------
start_time = time.time()
frame_count = 0
most_likely_gesture = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame,1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_results = face_mesh.process(rgb_frame)
    hand_results = hands.process(rgb_frame)

    blank_frame = np.zeros_like(frame)
    current_gesture = None

    # Draw face landmarks
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=blank_frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            if detect_smile(face_landmarks):
                current_gesture = "Smile"

    # Draw hand landmarks and detect gestures
    if hand_results.multi_hand_landmarks:
        hands_detected = len(hand_results.multi_hand_landmarks)
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(blank_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(255,255,255),thickness=2,circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(255,255,255),thickness=2))
        if hands_detected == 2:
            left, right = hand_results.multi_hand_landmarks
            if detect_heart(left,right):
                current_gesture = "Heart"
            elif detect_high_five(left,right):
                current_gesture = "High Five"
        for hand_landmarks in hand_results.multi_hand_landmarks:
            if detect_thumbs_up(hand_landmarks):
                current_gesture = "Thumbs Up"
            elif detect_victory_hand(hand_landmarks):
                current_gesture = "Victory Hand"
            elif detect_ok_hand(hand_landmarks):
                current_gesture = "OK Hand"
            elif face_results.multi_face_landmarks and detect_facepalm(hand_landmarks, face_results.multi_face_landmarks[0]):
                current_gesture = "Facepalm"

    if current_gesture:
        detection_history.append(current_gesture)

    # Determine most frequent gesture
    if detection_history:
        most_likely_gesture = Counter(detection_history).most_common(1)[0][0]
        if most_likely_gesture in gesture_emoji_map:
            set_floating_emoji(f"{most_likely_gesture} Detected", gesture_emoji_map[most_likely_gesture],
                               frame.shape[1]//2, frame.shape[0]//2)

    if floating_start_time and time.time() - floating_start_time > 1 and not detection_history:
        floating_emoji = None
        floating_text = None

    # Resize and center
    screen_height, screen_width = 720, 1280
    h, w, _ = blank_frame.shape
    scale = min(screen_width/w, screen_height/h)
    new_w, new_h = int(w*scale), int(h*scale)
    resized_frame = cv2.resize(blank_frame,(new_w,new_h))
    output_frame = np.zeros((screen_height,screen_width,3),dtype=np.uint8)
    x_offset = (screen_width - new_w)//2
    y_offset = (screen_height - new_h)//2
    output_frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_frame

    # Floating emoji
    if floating_emoji and time.time()-floating_start_time < 2:
        floating_emoji_y -= 10
        output_frame = draw_pil_text_on_cv2(output_frame,floating_text,floating_emoji,
                                            (floating_emoji_x,floating_emoji_y),text_font,emoji_font)

    # FPS
    frame_count +=1
    elapsed_time = time.time() - start_time
    if elapsed_time >1:
        fps = frame_count/elapsed_time
        frame_count =0
        start_time = time.time()
        print(f"FPS: {fps:.2f}, Detection: {detection_history}, Most Likely: {most_likely_gesture}")

    cv2.imshow("Gesture Detection", output_frame)
    if cv2.waitKey(1) & 0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()
