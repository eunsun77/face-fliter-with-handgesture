import mediapipe as mp
import cv2
import math
import numpy as np
import faceBlendCommon as fbc
import csv

# 핸드 제스처 변수
number = 0

# 필터 미리보기 이미지
black_bg = cv2.imread("filters/black_bg.jpg")

hand_1 = cv2.imread("filters/hand_1.png")
hand_2 = cv2.imread("filters/hand_2.png")
hand_3 = cv2.imread("filters/hand_3.png")
hand_4 = cv2.imread("filters/hand_4.png")
hand_5 = cv2.imread("filters/hand_5.png")

anonymous_preview = cv2.imread("filters/anonymous.png")
dog_preview = cv2.imread("filters/dog_preview.png")
cat_preview = cv2.imread("filters/cat_preview.png")
anime_preview = cv2.imread("filters/anime.png")
gold_crown_preview = cv2.imread("filters/gold-crown_preview.png")
emoji_preview = cv2.imread("filters/emoji.png")
heart_preview = cv2.imread("filters/heart.png")
smile_preview = cv2.imread("filters/smile.png")
sad_preview = cv2.imread("filters/sad.png")
spyderman_preview = cv2.imread("filters/spyderman.png")

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection

VISUALIZE_FACE_POINTS = False

filters_config = {
    'face_original':
        [{'path': "filters/face_original.png",
          'anno_path': "filters/face_original_annotations.csv",
          'morph': True, 'animated': False, 'has_alpha': True}],
    'anonymous1':
        [{'path': "filters/anonymous1.png",
          'anno_path': "filters/anonymous1_annotations.csv",
          'morph': True, 'animated': False, 'has_alpha': True}],
    'anonymous':
        [{'path': "filters/anonymous.png",
          'anno_path': "filters/anonymous_annotations.csv",
          'morph': True, 'animated': False, 'has_alpha': True}],
    'anime':
        [{'path': "filters/anime.png",
          'anno_path': "filters/anime_annotations.csv",
          'morph': True, 'animated': False, 'has_alpha': True}],
    'dog':
        [{'path': "filters/dog-ears.png",
          'anno_path': "filters/dog-ears_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True},
         {'path': "filters/dog-nose.png",
          'anno_path': "filters/dog-nose_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'cat':
        [{'path': "filters/cat-ears.png",
          'anno_path': "filters/cat-ears_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True},
         {'path': "filters/cat-nose.png",
          'anno_path': "filters/cat-nose_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'jason-joker':
        [{'path': "filters/jason-joker.png",
          'anno_path': "filters/jason-joker_annotations.csv",
          'morph': True, 'animated': False, 'has_alpha': True}],
    'gold-crown':
        [{'path': "filters/gold-crown.png",
          'anno_path': "filters/gold-crown_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'flower-crown':
        [{'path': "filters/flower-crown.png",
          'anno_path': "filters/flower-crown_annotations.csv",
          'morph': False, 'animated': False, 'has_alpha': True}],
    'emoji':
        [{'path': "filters/emoji.png",
          'anno_path': "filters/emoji_annotations.csv",
          'morph': True, 'animated': False, 'has_alpha': True}],
    'heart':
        [{'path': "filters/heart.png",
          'anno_path': "filters/heart_annotations.csv",
          'morph': True, 'animated': False, 'has_alpha': True}],
    'smile':
        [{'path': "filters/smile.png",
          'anno_path': "filters/smile_annotations.csv",
          'morph': True, 'animated': False, 'has_alpha': True}],
    'sad':
        [{'path': "filters/sad.png",
          'anno_path': "filters/sad_annotations.csv",
          'morph': True, 'animated': False, 'has_alpha': True}],
    'spyderman':
        [{'path': "filters/spyderman.png",
          'anno_path': "filters/spyderman_annotations.csv",
          'morph': True, 'animated': False, 'has_alpha': True}],
}


# 이미지에서 얼굴의 랜드마크 탐지 detect facial landmarks in image
def getLandmarks(img):
    mp_face_mesh = mp.solutions.face_mesh
    selected_keypoint_indices = [127, 93, 58, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 288, 323, 356, 70, 63, 105, 66, 55,
                 285, 296, 334, 293, 300, 168, 6, 195, 4, 64, 60, 94, 290, 439, 33, 160, 158, 173, 153, 144, 398, 385,
                 387, 466, 373, 380, 61, 40, 39, 0, 269, 270, 291, 321, 405, 17, 181, 91, 78, 81, 13, 311, 306, 402, 14,
                 178, 162, 54, 67, 10, 297, 284, 389]

    height, width = img.shape[:-1]

    with mp_face_mesh.FaceMesh(max_num_faces=1, static_image_mode=True, min_detection_confidence=0.5) as face_mesh:

        results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if not results.multi_face_landmarks:
            print('Face not detected!!!')
            return 0

        for face_landmarks in results.multi_face_landmarks:
            values = np.array(face_landmarks.landmark)
            face_keypnts = np.zeros((len(values), 2))

            for idx,value in enumerate(values):
                face_keypnts[idx][0] = value.x
                face_keypnts[idx][1] = value.y

            # Convert normalized points to image coordinates
            face_keypnts = face_keypnts * (width, height)
            face_keypnts = face_keypnts.astype('int')

            relevant_keypnts = []

            for i in selected_keypoint_indices:
                relevant_keypnts.append(face_keypnts[i])
            return relevant_keypnts
    return 0


def load_filter_img(img_path, has_alpha):
    # Read the image
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    alpha = None
    if has_alpha:
        b, g, r, alpha = cv2.split(img)
        img = cv2.merge((r, g, b))

    return img, alpha

def load_landmarks(annotation_file):
    with open(annotation_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        points = {}
        for i, row in enumerate(csv_reader):
            # skip head or empty line if it's there
            try:
                x, y = int(row[1]), int(row[2])
                points[row[0]] = (x, y)
            except ValueError:
                continue
        return points


def find_convex_hull(points):
    hull = []
    hullIndex = cv2.convexHull(np.array(list(points.values())), clockwise=False, returnPoints=False)
    addPoints = [
        [48], [49], [50], [51], [52], [53], [54], [55], [56], [57], [58], [59],  # Outer lips
        [60], [61], [62], [63], [64], [65], [66], [67],  # Inner lips
        [27], [28], [29], [30], [31], [32], [33], [34], [35],  # Nose
        [36], [37], [38], [39], [40], [41], [42], [43], [44], [45], [46], [47],  # Eyes
        [17], [18], [19], [20], [21], [22], [23], [24], [25], [26]  # Eyebrows
    ]
    hullIndex = np.concatenate((hullIndex, addPoints))
    for i in range(0, len(hullIndex)):
        hull.append(points[str(hullIndex[i][0])])

    return hull, hullIndex


def load_filter(filter_name="dog"):

    filters = filters_config[filter_name]

    multi_filter_runtime = []

    for filter in filters:
        temp_dict = {}

        img1, img1_alpha = load_filter_img(filter['path'], filter['has_alpha'])

        temp_dict['img'] = img1
        temp_dict['img_a'] = img1_alpha

        points = load_landmarks(filter['anno_path'])

        temp_dict['points'] = points

        if filter['morph']:
            # Find convex hull for delaunay triangulation using the landmark points
            hull, hullIndex = find_convex_hull(points)

            # Find Delaunay triangulation for convex hull points
            sizeImg1 = img1.shape
            rect = (0, 0, sizeImg1[1], sizeImg1[0])
            dt = fbc.calculateDelaunayTriangles(rect, hull)

            temp_dict['hull'] = hull
            temp_dict['hullIndex'] = hullIndex
            temp_dict['dt'] = dt

            if len(dt) == 0:
                continue

        if filter['animated']:
            filter_cap = cv2.VideoCapture(filter['path'])
            temp_dict['cap'] = filter_cap

        multi_filter_runtime.append(temp_dict)

    return filters, multi_filter_runtime

# Initialize MediaPipe Hands and Face Detection
hands = mp_hands.Hands(max_num_hands=2)
face_detection = mp_face_detection.FaceDetection()

# Set up a VideoCapture object
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

count = 0
isFirstFrame = True
sigma = 50

iter_filter_keys = iter(filters_config.keys())
filters, multi_filter_runtime = load_filter(next(iter_filter_keys))

while True:
    # Read the frame from the video capture object
    ret, frame = cap.read()

    # Convert the image to RGB format
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Use MediaPipe Face Detection to detect faces in the image
    results = face_detection.process(frame)

    # If a face is detected, apply the mask filter around the face
    if not ret:
        break
    else:

        points2 = getLandmarks(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # if face is partially detected
        if not points2 or (len(points2) != 75):
            continue

        ################ Optical Flow and Stabilization Code #####################
        img2Gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if isFirstFrame:
            points2Prev = np.array(points2, np.float32)
            img2GrayPrev = np.copy(img2Gray)
            isFirstFrame = False

        lk_params = dict(winSize=(101, 101), maxLevel=15,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.001))
        points2Next, st, err = cv2.calcOpticalFlowPyrLK(img2GrayPrev, img2Gray, points2Prev,
                                                        np.array(points2, np.float32),
                                                        **lk_params)

        # Final landmark points are a weighted average of detected landmarks and tracked landmarks

        for k in range(0, len(points2)):
            d = cv2.norm(np.array(points2[k]) - points2Next[k])
            alpha = math.exp(-d * d / sigma)
            points2[k] = (1 - alpha) * np.array(points2[k]) + alpha * points2Next[k]
            points2[k] = fbc.constrainPoint(points2[k], frame.shape[1], frame.shape[0])
            points2[k] = (int(points2[k][0]), int(points2[k][1]))

        # Update variables for next pass
        points2Prev = np.array(points2, np.float32)
        img2GrayPrev = img2Gray
        ################ End of Optical Flow and Stabilization Code ###############

        if VISUALIZE_FACE_POINTS:
            for idx, point in enumerate(points2):
                cv2.circle(frame, point, 2, (255, 0, 0), -1)
                cv2.putText(frame, str(idx), point, cv2.FONT_HERSHEY_SIMPLEX, .3, (255, 255, 255), 1)
            cv2.imshow("landmarks", frame)

        for idx, filter in enumerate(filters):

            filter_runtime = multi_filter_runtime[idx]
            img1 = filter_runtime['img']
            points1 = filter_runtime['points']
            img1_alpha = filter_runtime['img_a']

            if filter['morph']:

                hullIndex = filter_runtime['hullIndex']
                dt = filter_runtime['dt']
                hull1 = filter_runtime['hull']

                # create copy of frame
                warped_img = np.copy(frame)

                # Find convex hull
                hull2 = []
                for i in range(0, len(hullIndex)):
                    hull2.append(points2[hullIndex[i][0]])

                mask1 = np.zeros((warped_img.shape[0], warped_img.shape[1]), dtype=np.float32)
                mask1 = cv2.merge((mask1, mask1, mask1))
                img1_alpha_mask = cv2.merge((img1_alpha, img1_alpha, img1_alpha))

                # Warp the triangles
                for i in range(0, len(dt)):
                    t1 = []
                    t2 = []

                    for j in range(0, 3):
                        t1.append(hull1[dt[i][j]])
                        t2.append(hull2[dt[i][j]])

                    fbc.warpTriangle(img1, warped_img, t1, t2)
                    fbc.warpTriangle(img1_alpha_mask, mask1, t1, t2)

                # Blur the mask before blending (블렌딩 하기 전에 마스크 블러처리)
                mask1 = cv2.GaussianBlur(mask1, (3, 3), 10)

                mask2 = (255.0, 255.0, 255.0) - mask1

                # Perform alpha blending of the two images (두 이미지의 알파 블렌딩 수행)
                temp1 = np.multiply(warped_img, (mask1 * (1.0 / 255)))
                temp2 = np.multiply(frame, (mask2 * (1.0 / 255)))
                output = temp1 + temp2
            else:
                dst_points = [points2[int(list(points1.keys())[0])], points2[int(list(points1.keys())[1])]]
                tform = fbc.similarityTransform(list(points1.values()), dst_points)
                # Apply similarity transform to input image (입력 이미지에 similarity transform 적용)
                trans_img = cv2.warpAffine(img1, tform, (frame.shape[1], frame.shape[0]))
                trans_alpha = cv2.warpAffine(img1_alpha, tform, (frame.shape[1], frame.shape[0]))
                mask1 = cv2.merge((trans_alpha, trans_alpha, trans_alpha))

                # Blur the mask before blending (블렌딩 하기 전에 마스크 블러처리)
                mask1 = cv2.GaussianBlur(mask1, (3, 3), 10)

                mask2 = (255.0, 255.0, 255.0) - mask1

                # Perform alpha blending of the two images (두 이미지의 알파 블렌딩 수행)
                temp1 = np.multiply(trans_img, (mask1 * (1.0 / 255)))
                temp2 = np.multiply(frame, (mask2 * (1.0 / 255)))
                output = temp1 + temp2

            frame = output = np.uint8(output)
            
        keypressed = cv2.waitKey(1) & 0xFF
        if keypressed == 27:
            break
            
            # 아래 주석 처리된 부분은 'f' 버튼을 누르면 필터가 바뀌는 방식
            # Put next filter if 'f' is pressed
            # elif keypressed == ord('f'):
                # try:
                    # filters, multi_filter_runtime = load_filter(next(iter_filter_keys))
                # except:
                    # fiter_filter_keys = iter(filters_config.keys())
                    # filters, multi_filter_runtime = load_filter(next(iter_filter_keys))
            
            # elif number == 1:
                # try:        
                    # filters, multi_filter_runtime = load_filter(next(iter_filter_keys))
                # except:
                    # fiter_filter_keys = iter(filters_config.keys())
                    # filters, multi_filter_runtime = load_filter(next(iter_filter_keys))
                    
            # count += 1
            
        # 핸드 제스처가 인식되면 number에 저장되고 number에 맞는 필터를 씌움
        if number == 0:
            filters, multi_filter_runtime = load_filter("face_original")
                    
        elif number == 1:
            filters, multi_filter_runtime = load_filter("anonymous")
                
        elif number == 2:
            filters, multi_filter_runtime = load_filter("dog")
                
        elif number == 3:
            filters, multi_filter_runtime = load_filter("cat")
                
        elif number == 4:
            filters, multi_filter_runtime = load_filter("anime")
                
        elif number == 5:
            filters, multi_filter_runtime = load_filter("gold-crown")

    # Use MediaPipe Hands to detect hands in the image (MediaPipe Hands를 사용하여 손 감지)
    results = hands.process(frame)

    # If hands are detected, draw landmarks and connections on the image (손이 감지되면 이미지에 랜드마크 및 연결선을 그림)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # hand skeleton을 출력하고 싶은 경우 밑줄을 주석 해제하고 continue를 주석 처리
            # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            continue
            
        landmark_list = []
        for idx, landmark in enumerate(hand_landmarks.landmark):
            h, w, c = frame.shape
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            landmark_list.append((cx, cy))
          
        # 핸드 제스처 0 ~ 5까지
        if(landmark_list[4][1] > landmark_list[5][1] and landmark_list[8][1] > landmark_list[6][1] and landmark_list[12][1] > landmark_list[10][1] and landmark_list[16][1] > landmark_list[14][1] and landmark_list[20][1] > landmark_list[18][1]):
            number = 0
        
        elif(landmark_list[4][1] > landmark_list[5][1] and landmark_list[8][1] < landmark_list[6][1] and landmark_list[12][1] > landmark_list[10][1] and landmark_list[16][1] > landmark_list[14][1] and landmark_list[20][1] > landmark_list[18][1]):
            number = 1
          
        elif(landmark_list[4][1] > landmark_list[5][1] and landmark_list[8][1] < landmark_list[6][1] and landmark_list[12][1] < landmark_list[10][1] and landmark_list[16][1] > landmark_list[14][1] and landmark_list[20][1] > landmark_list[18][1]):
            number = 2
          
        elif(landmark_list[4][1] > landmark_list[5][1] and landmark_list[8][1] < landmark_list[6][1] and landmark_list[12][1] < landmark_list[10][1] and landmark_list[16][1] < landmark_list[14][1] and landmark_list[20][1] > landmark_list[18][1]):
            number = 3
          
        elif(landmark_list[4][1] > landmark_list[5][1] and landmark_list[8][1] < landmark_list[6][1] and landmark_list[12][1] < landmark_list[10][1] and landmark_list[16][1] < landmark_list[14][1] and landmark_list[20][1] < landmark_list[18][1]):
            number = 4
          
        elif(landmark_list[4][1] < landmark_list[5][1] and landmark_list[8][1] < landmark_list[6][1] and landmark_list[12][1] < landmark_list[10][1] and landmark_list[16][1] < landmark_list[14][1] and landmark_list[20][1] < landmark_list[18][1]):
            number = 5
    
    # 카메라가 FHD 해상도를 지원하면 주석처리
    frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_LANCZOS4)
    
    # 카메라 좌우반전
    frame = cv2.flip(frame, 1)
    
    # 미리보기 이미지의 크기를 변경
    black_bg_resized = cv2.resize(black_bg, (270, 1080))
    
    hand_1_resized = cv2.resize(hand_1, (100, 150))
    anonymous_preview_resized = cv2.resize(anonymous_preview, (150, 150))
    anonymous_preview_resized = cv2.cvtColor(anonymous_preview_resized, cv2.COLOR_BGR2RGB)
    
    hand_2_resized = cv2.resize(hand_2, (100, 150))
    dog_preview_resized = cv2.resize(dog_preview, (150, 150))
    dog_preview_resized = cv2.cvtColor(dog_preview_resized, cv2.COLOR_BGR2RGB)
    
    hand_3_resized = cv2.resize(hand_3, (100, 150))
    cat_preview_resized = cv2.resize(cat_preview, (150, 150))
    cat_preview_resized = cv2.cvtColor(cat_preview_resized, cv2.COLOR_BGR2RGB)
    
    hand_4_resized = cv2.resize(hand_4, (100, 150))
    anime_preview_resized = cv2.resize(anime_preview, (150, 150))
    anime_preview_resized = cv2.cvtColor(anime_preview_resized, cv2.COLOR_BGR2RGB)
    
    hand_5_resized = cv2.resize(hand_5, (100, 150))
    gold_crown_preview_resized = cv2.resize(gold_crown_preview, (150, 150))
    gold_crown_preview_resized = cv2.cvtColor(gold_crown_preview_resized, cv2.COLOR_BGR2RGB)
    
    # 미리보기 이미지를 화면 우측에 넣기
    frame[0:1080, 1650:1920] = black_bg_resized
    
    frame[65:215, 1660:1760] = hand_1_resized
    frame[65:215, 1760:1910] = anonymous_preview_resized
    
    frame[265:415, 1660:1760] = hand_2_resized
    frame[265:415, 1760:1910] = dog_preview_resized
    
    frame[465:615, 1660:1760] = hand_3_resized
    frame[465:615, 1760:1910] = cat_preview_resized
    
    frame[665:815, 1660:1760] = hand_4_resized
    frame[665:815, 1760:1910] = anime_preview_resized
    
    frame[865:1015, 1660:1760] = hand_5_resized
    frame[865:1015, 1760:1910] = gold_crown_preview_resized
    
    if(number == 0):
        print(f"제스처: {number}, 필터: face_original")
        
    elif(number == 1):
        print(f"제스처: {number}, 필터: anonymous")
        
    elif(number == 2):
        print(f"제스처: {number}, 필터: dog")
        
    elif(number == 3):
        print(f"제스처: {number}, 필터: cat")
        
    elif(number == 4):
        print(f"제스처: {number}, 필터: anime")
        
    elif(number == 5):
        print(f"제스처: {number}, 필터: crown")
    
    # Convert the image back to BGR format and display the image (이미지를 다시 BGR 형식으로 변환하고 이미지를 표시)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    cv2.putText(frame, f"Now: {number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("MediaPipe Hands and Face Detection", frame)
    
    # Exit the loop if 'q' is pressed ('q'를 누르면 루프를 종료)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the resources and close the window (리소스를 해제하고 창을 닫음)
cap.release()
cv2.destroyAllWindows()