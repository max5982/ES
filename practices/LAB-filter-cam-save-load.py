import cv2, json, os
from datetime import datetime
import numpy as np

def nothing(x): pass

json_name = "LAB-cal.json"

# 1. 예외처리로 config 읽기
def load_lab_config(json_name):
    # 기본값 정의
    default_config = {
        "l_min": 0, "l_max": 255,
        "a_min": 0, "a_max": 255,
        "b_min": 0, "b_max": 255
    }
    try:
        with open(json_name) as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"'{json_name}' 파일이 없습니다. 기본값을 사용합니다.")
        config = default_config
    except json.JSONDecodeError:
        print(f"'{json_name}' 파일이 손상됐거나 올바른 JSON이 아닙니다. 기본값을 사용합니다.")
        config = default_config
    except Exception as e:
        print(f"예상치 못한 오류: {e}. 기본값을 사용합니다.")
        config = default_config
    return config

config = load_lab_config(json_name)

l_min = config["l_min"]
l_max = config["l_max"]
a_min = config["a_min"]
a_max = config["a_max"]
b_min = config["b_min"]
b_max = config["b_max"]

# 트랙바 창 생성
cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars", 800, 260)
cv2.createTrackbar("L Min", "Trackbars", l_min, 255, nothing)
cv2.createTrackbar("L Max", "Trackbars", l_max, 255, nothing)
cv2.createTrackbar("A Min", "Trackbars", a_min, 255, nothing)
cv2.createTrackbar("A Max", "Trackbars", a_max, 255, nothing)
cv2.createTrackbar("B Min", "Trackbars", b_min, 255, nothing)
cv2.createTrackbar("B Max", "Trackbars", b_max, 255, nothing)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret: break
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    # 트랙바 값 읽기
    l_min = cv2.getTrackbarPos("L Min", "Trackbars")
    l_max = cv2.getTrackbarPos("L Max", "Trackbars")
    a_min = cv2.getTrackbarPos("A Min", "Trackbars")
    a_max = cv2.getTrackbarPos("A Max", "Trackbars")
    b_min = cv2.getTrackbarPos("B Min", "Trackbars")
    b_max = cv2.getTrackbarPos("B Max", "Trackbars")

    lower = np.array([l_min, a_min, b_min])
    upper = np.array([l_max, a_max, b_max])
    mask = cv2.inRange(lab, lower, upper)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow("LAB Filter", cv2.resize(res, (640,480)))
    key = cv2.waitKey(1)

    if key==27:
        break
    elif key==ord('s'):
        # 변수들을 다시 딕셔너리로 묶어서 저장
        save_data = {
            "l_min": l_min,
            "l_max": l_max,
            "a_min": a_min,
            "a_max": a_max,
            "b_min": b_min,
            "b_max": b_max
        }
        now = datetime.now().strftime("%y%m%d-%H%M%S")

        with open(f"LAB-cal-{now}.json","w") as f:
            json.dump(save_data, f,indent=4)
        with open(json_name,"w") as f:
            json.dump(save_data, f,indent=4)        

cap.release()
cv2.destroyAllWindows()
