import streamlit as st
import cv2
import numpy as np
import time

date = True
day = True
st.title("Motion Detector")
st.set_page_config(page_title="Motion Detector", page_icon=":camera:")
with st.expander("Start Camera"):
    streamlit_image = st.image([])
    camera = cv2.VideoCapture(0)

kernel = np.ones((3, 3), np.uint8)
first_frame = None

while True:
    check, frame = camera.read()
    if date:
        date = time.strftime("%H:%M.%S")
        day = time.strftime("%A")

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    bw_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    bw_gau_frame = cv2.GaussianBlur(bw_frame, (21, 21), 0)

    if first_frame is None:
        first_frame = bw_gau_frame

    delta_frame = cv2.absdiff(first_frame, bw_gau_frame)
    thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    dilated_frame = cv2.dilate(thresh_frame, None,  iterations = 3)

    contours, check = cv2.findContours(dilated_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 5000:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        rectangle = cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 3)

    cv2.putText(img=frame, text=day, org=(50,50), fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=2, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    clock = cv2.putText(img=frame, text=date, org=(50,100), fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=2, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)




    streamlit_image.image(frame)



