import os
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
from tensorflow.keras.utils import load_img
from tensorflow.keras.preprocessing import image
from keras.models import load_model
import streamlit as st
from PIL import Image
import tempfile
import subprocess
import io
import av

def main():


             
             

    age_model = load_model(r"C:/Users/Nidhal Jegham/Facial Recognition Project/Age Prediction Model.h5",
                                       compile=True
                                       )


    gender_model = load_model(r"C:/Users/Nidhal Jegham/Facial Recognition Project/Gender Prediction Model.h5",
                                       safe_mode=False
                                       )

    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    age_classes=['1-2','3-9','10-20','21-27','28-45','46-65','+66']
    gender_classes=['Man','Woman']
    race_classes=['1',"2","3","4","h5"]
    
    st.title("""

              Facial Recognition App
             
             """)
             
    
    
    st.markdown(
        """
        <style>
        [data_testid= "stSidebar"][aria_expanded="true"]> div: first-child{
            
            width=350px
            }
        [data_testid= "stSidebar"][aria_expanded="false"]> div: first-child{
            
            width=350px
            margin-left: -350px
            }
        <style>
        
        """,
        unsafe_allow_html=True,
        
        )
    
    st.sidebar.title('Face Recognition App Sidebar')
    st.sidebar.subheader('Parameters')
    
    app_mode=st.sidebar.selectbox("Select The App Mode", [ "Image Recognition", "Video Recognition"])
    
    st.sidebar.markdown('------')

    
    image_scale= st.sidebar.slider("Image Scale Factor", 1.2, 5.0, 1.2, 0.1)
    
    st.sidebar.markdown('------')
    gender_scale= st.sidebar.slider("Gender Confidence Factor",0.01, 0.99, 0.9, 0.01)
    
    st.sidebar.markdown('------')

    age_scale= st.sidebar.slider("Age Confidence Factor", value=0.85, min_value=0.01, max_value=0.99)


    
    uploaded_file= st.sidebar.file_uploader('Upload your file')
 
    
    demo_picture=r"C:\Users\Nidhal Jegham\Facial Recognition Project\Pictures\Leo3.jpg"
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

            

    def shrink_face_roi(x, y, w, h, scale):
        wh_multiplier = (1-scale)/2
        x_new = int(x + (w * wh_multiplier))
        y_new = int(y + (h * wh_multiplier))
        w_new = int(w * scale)
        h_new = int(h * scale)
        return (x_new, y_new, w_new, h_new)


    def create_age_text(img, face_age, face_gender, x, y, w, h):
        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        text_scale = 1
        age_scale = 0.8
    
        
        gender_text_width=0
        gender_text_height=0
        age_text_width=0
        age_text_height=0
        for i in range(6,30):
            (gender_text_width, gender_text_height), gender_text_bsln = cv2.getTextSize(face_gender, fontFace=fontFace, fontScale=i/10, thickness=2)
            if gender_text_width>= w//4:
                text_scale=(i-1)/10
                gender_text_width=gender_text_width
                gender_text_height=gender_text_height
                break
    
        for i in range(6,30):
            (age_text_width, age_text_height), age_text_bsln = cv2.getTextSize(face_age, fontFace=fontFace, fontScale=i/10, thickness=1)
            if age_text_width>= (w//18)*9:
                age_text_width=age_text_width
                age_text_height=age_text_height
                age_scale=(i-1)/10
                break
    
    
    
    
        x_gender_text_org = x+w//56
        y_gender_text_org = int((y + gender_text_height)*1.1)-gender_text_height//3
    
        x_age_org = x + w -age_text_width
        y_age_org = int(y+h)- int(age_text_height*0.5)
    
        gender_rect_top_left = (x, y)
        gender_rect_bottom_right = ((int((x + gender_text_width)*1.02), int((y + gender_text_height)*1.1)))
        age_rect_top_left = (x + w - int(age_text_width*1.1) , y+h-age_text_height*2)
        age_rect_bottom_right = (x + w , int((y+h)))
    
        face_age_background = cv2.rectangle(img, gender_rect_top_left, gender_rect_bottom_right, (0, 0, 102), cv2.FILLED)
        face_age_background = cv2.rectangle(img, age_rect_top_left, age_rect_bottom_right, (0, 0, 102), cv2.FILLED)
    
        face_gender_text = cv2.putText(img, face_gender, org=(x_gender_text_org, y_gender_text_org), fontFace=fontFace,
                                       fontScale=text_scale, thickness=2, color=(255, 255, 255), lineType=cv2.LINE_AA)
        face_age_text = cv2.putText(img, face_age, org=(x_age_org, y_age_org), fontFace=fontFace, fontScale=age_scale,
                                    thickness=2, color=(255, 255, 255), lineType=cv2.LINE_AA)
       
        return (face_age_background, face_age_text, face_gender_text)
    
    
    
    def classify_age(img):
    
        img_copy = np.copy(img)
        img_copy = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_copy_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_copy_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    
        faces = face_cascade.detectMultiScale(img_copy, scaleFactor=image_scale, minNeighbors=6, minSize=(100, 100))
    
        for i, (x, y, w, h) in enumerate(faces):
    
            face_rect = cv2.rectangle(img_copy, (x, y), (x+w, int((y+h)*1.2)), (0, 0, 102), thickness=4)
            x2, y2, w2, h2 = shrink_face_roi(x, y-20, w, h, gender_scale)
            x3, y3, w3, h3 = shrink_face_roi(x, y-20, w, h, age_scale)
    
            face_roi_rgb = img_copy_rgb[y2:y2+h2, x2:x2+w2]
            face_roi_gray = img_copy_gray[y3:y3+h3, x3:x3+w3]
            face_roi_rgb_gender = cv2.resize(face_roi_rgb, (218, 178))
            face_roi_rgb_race = cv2.resize(face_roi_rgb, (148, 148))
            face_roi_gray = cv2.resize(face_roi_gray, (200, 200))
    
    
            features_rgb_gender = []
            face_roi_rgb_gender = np.array(face_roi_rgb_gender)
            face_roi_rgb_gender=face_roi_rgb_gender/255
            features_rgb_gender.append(face_roi_rgb_gender)
            features_rgb_gender=np.array(features_rgb_gender)
    
            features_rgb_race = []
            face_roi_rgb_race = np.array(face_roi_rgb_race)
            face_roi_rgb_race=face_roi_rgb_race/255
            features_rgb_race.append(face_roi_rgb_race)
            features_rgb_race=np.array(features_rgb_race)
    
            features_gray = []
            face_roi_gray = np.array(face_roi_gray)
            face_roi_rgb=face_roi_gray/255
            features_gray.append(face_roi_gray)
            features_gray=np.array(features_gray)
    
            face_age = age_classes[np.argmax(age_model.predict(features_gray))]
            face_age= face_age+' Years Old'
            face_gender = gender_classes[np.argmax(gender_model.predict(features_rgb_gender))]
            

    
            face_age_background, face_age_text, face_gender = create_age_text(img_copy, face_age, face_gender, x, y, w, (int((y+h)*1.2))-y)
    
        return img_copy
    
    def confidence_mes(img):
        img_copy = np.copy(img)
        img_copy = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_copy_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_copy_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    
        faces = face_cascade.detectMultiScale(img_copy, scaleFactor=1.2, minNeighbors=6, minSize=(100, 100))
        num_face= len(faces)
    
        for i, (x, y, w, h) in enumerate(faces):
    
            face_rect = cv2.rectangle(img_copy, (x, y), (x+w, int((y+h)*1.2)), (0, 0, 102), thickness=4)
            x2, y2, w2, h2 = shrink_face_roi(x, y-20, w, h, 0.9)
            x3, y3, w3, h3 = shrink_face_roi(x, y-20, w, h, 0.85)
    
            face_roi_rgb = img_copy_rgb[y2:y2+h2, x2:x2+w2]
            face_roi_gray = img_copy_gray[y3:y3+h3, x3:x3+w3]
            face_roi_rgb_gender = cv2.resize(face_roi_rgb, (218, 178))
            face_roi_rgb_race = cv2.resize(face_roi_rgb, (148, 148))
            face_roi_gray = cv2.resize(face_roi_gray, (200, 200))
    
    
            features_rgb_gender = []
            face_roi_rgb_gender = np.array(face_roi_rgb_gender)
            face_roi_rgb_gender=face_roi_rgb_gender/255
            features_rgb_gender.append(face_roi_rgb_gender)
            features_rgb_gender=np.array(features_rgb_gender)
    
            features_rgb_race = []
            face_roi_rgb_race = np.array(face_roi_rgb_race)
            face_roi_rgb_race=face_roi_rgb_race/255
            features_rgb_race.append(face_roi_rgb_race)
            features_rgb_race=np.array(features_rgb_race)
    
            features_gray = []
            face_roi_gray = np.array(face_roi_gray)
            face_roi_rgb=face_roi_gray/255
            features_gray.append(face_roi_gray)
            features_gray=np.array(features_gray)
            age_conf_list=[]
            gender_conf_list=[]
            
            age_conf_list.append(np.max(age_model.predict(features_gray)))
            gender_conf_list.append(np.max(gender_model.predict(features_rgb_gender)))


            
    
            
        conf_age=sum(np.array(age_conf_list))/num_face
        conf_gender=sum(np.array(gender_conf_list))/num_face
        return num_face, "{:.3f}".format(conf_age), "{:.3f}".format(conf_gender)


    
    

    
    if uploaded_file is None:
        
        if app_mode=='Image Recognition':
        
            img = cv2.imread(demo_picture)
            age_img = classify_age(img)
            num_face, age_conf, gender_conf= confidence_mes(img)
            
            st.image(age_img, caption='')
            
           



        if app_mode=="Video Recognition":
        
        
        
            def classify_age_demo(img):
            
                img_copy = np.copy(img)
                img_copy = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_copy_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_copy_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            
                faces = face_cascade.detectMultiScale(img_copy, scaleFactor=image_scale, minNeighbors=6, minSize=(100, 100))
            
                for i, (x, y, w, h) in enumerate(faces):
            
                    face_rect = cv2.rectangle(img_copy, (x, y), (x+w, int((y+h)*1.2)), (0, 0, 102), thickness=4)
                    x2, y2, w2, h2 = shrink_face_roi(x, y-20, w, h, gender_scale)
                    x3, y3, w3, h3 = shrink_face_roi(x, y-20, w, h, age_scale)
            
                    face_roi_rgb = img_copy_rgb[y2:y2+h2, x2:x2+w2]
                    face_roi_gray = img_copy_gray[y3:y3+h3, x3:x3+w3]
                    face_roi_rgb_gender = cv2.resize(face_roi_rgb, (218, 178))
                    face_roi_rgb_race = cv2.resize(face_roi_rgb, (148, 148))
                    face_roi_gray = cv2.resize(face_roi_gray, (200, 200))
            
            
                    features_rgb_gender = []
                    face_roi_rgb_gender = np.array(face_roi_rgb_gender)
                    face_roi_rgb_gender=face_roi_rgb_gender/255
                    features_rgb_gender.append(face_roi_rgb_gender)
                    features_rgb_gender=np.array(features_rgb_gender)
            
                    features_rgb_race = []
                    face_roi_rgb_race = np.array(face_roi_rgb_race)
                    face_roi_rgb_race=face_roi_rgb_race/255
                    features_rgb_race.append(face_roi_rgb_race)
                    features_rgb_race=np.array(features_rgb_race)
            
                    features_gray = []
                    face_roi_gray = np.array(face_roi_gray)
                    face_roi_rgb=face_roi_gray/255
                    features_gray.append(face_roi_gray)
                    features_gray=np.array(features_gray)
            
                    face_age = age_classes[np.argmax(age_model.predict(features_gray))]
                    face_age= "21-27"+' Years Old'
                    face_gender = gender_classes[np.argmax(gender_model.predict(features_rgb_gender))]
                    
    
            
                    face_age_background, face_age_text, face_gender = create_age_text(img_copy, face_age, face_gender, x, y, w, (int((y+h)*1.2))-y)
            
                return img_copy

            video_path=r"C:\Users\Nidhal Jegham\Facial Recognition Project\Pictures\Nidhal Video.mp4"
            stframe=st.empty()
            tfile = tempfile.NamedTemporaryFile(delete=False)
            vid = cv2.VideoCapture(video_path)

            frame_width = int(vid.get(3))
            frame_height = int(vid.get(4))
            fps_input=int(vid.get(cv2.CAP_PROP_FPS))
            
            codec= cv2.VideoWriter_fourcc('m',"p","4","v")
            out=cv2.VideoWriter("output_1.mp4", codec, fps_input, (frame_width,frame_height))


                            
            
            while(vid.isOpened()):
                
                ret, frame = vid.read()
                
                if ret==True:
                    
                    anz_img = classify_age_demo(frame)
                    num_face, age_conf, gender_conf= confidence_mes(frame)
                    

                   

                   
                    out.write(anz_img)
            
                else:
                    break
                stframe.image(anz_img)
                
            out.release()
            vid.release()
            
            
        st.write("Welcome to my first <b><span style='color:red'>Deep Learning</span></b> project. In this project, I've developed models to classify <b><span style='color:blue'>gender</span></b> and predict <b><span style='color:green'>age</span></b> based on images and videos.", unsafe_allow_html=True)
        st.write("This project helped me learn and apply various techniques in data preprocessing, data augmentation, model building, and visualization. The models have reached an accuracy of <span style='color:green'>***%96***</span> classifying gender and <span style='color:green'>***%84***</span> classifying age.", unsafe_allow_html=True)
        st.write("Feel free to explore the features and functionalities of this project. Upload your images or videos, and let the models do their magic! Don't hesitate to provide feedback or suggestions for improvement. Thank you for visiting, and I hope you enjoy your time here!")
        
        st.markdown("You can check out my other projects on my [Github account](https://github.com/Nidhal-Jegham/) or [Kaggle account](https://www.kaggle.com/nidhaljegham).")














    
    
    
    
    
    
    
    
    
    
    
    if uploaded_file is not None :
        list_path=[]
        list_path=uploaded_file.name.split('.')
        path_type=list_path[-1]
        if path_type in['png', 'jpg']:
            path= Image.open(uploaded_file)
            cap=np.array(path)
            cv2.imwrite('temp.jpg', cv2.cvtColor(cap, cv2.COLOR_RGB2BGR))
            img = cv2.imread('temp.jpg')
            age_img = classify_age(img)
            num_face, age_conf, gender_conf= confidence_mes(img)
            
            st.image(age_img, caption='')
            kpi1, kpi2, kpi3= st.columns(3)
            with kpi1:
                kpi1_markdown=st.markdown("**Number of Faces**")
                kpi1_text= st.markdown('0')
                
            with kpi2:
                kpi2_markdown=st.markdown("**Gender Confidence**")
                kpi2_text= st.markdown('0') 
                
            with kpi3:
                kpi3_markdown=st.markdown("**Age Confidence**")
                kpi3_text= st.markdown('0')
        
            kpi1_markdown.write(f"<h1 style='text-align: center;font-weight:normal; color:#bd4043; font-size:20px'>Number of Faces</h1>", unsafe_allow_html=True)
            kpi1_text.write(f"<h1 style='text-align: center;font-weight:normal; color:white; font-size:18px'>{num_face}</h1>", unsafe_allow_html=True)
            
            kpi2_markdown.write(f"<h1 style='text-align: center;font-weight:normal; color:#bd4043; font-size:20px'>Gender Confidence</h1>", unsafe_allow_html=True)
            kpi2_text.write(f"<h1 style='text-align: center;font-weight:normal; color:white; font-size:18px'>{gender_conf}</h1>", unsafe_allow_html=True)
            
            kpi3_markdown.write(f"<h1 style='text-align: center;font-weight:normal; color:#bd4043; font-size:20px'>Age Confidence</h1>", unsafe_allow_html=True)
            kpi3_text.write(f"<h1 style='text-align: center;font-weight:normal; color:white; font-size:18px'>{age_conf}</h1>", unsafe_allow_html=True)
            

            
            
            
            
            
        if path_type=="mp4":
            

            stframe=st.empty()
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            vid = cv2.VideoCapture(tfile.name)

            frame_width = int(vid.get(3))
            frame_height = int(vid.get(4))
            fps_input=int(vid.get(cv2.CAP_PROP_FPS))
            
            codec= cv2.VideoWriter_fourcc('m',"p","4","v")
            out=cv2.VideoWriter("output_1.mp4", codec, fps_input, (frame_width,frame_height))
            kpi1, kpi2, kpi3= st.columns(3)
            
            
            with kpi1:
                kpi1_markdown=st.markdown("**Number of Faces**")
                kpi1_text= st.markdown('0')
                        
            with kpi2:
                kpi2_markdown=st.markdown("**Gender Confidence**")
                kpi2_text= st.markdown('0') 
                        
            with kpi3:
                kpi3_markdown=st.markdown("**Age Confidence**")
                kpi3_text= st.markdown('0')
                            
            
            while(vid.isOpened()):
                
                ret, frame = vid.read()
                
                if ret==True:
                    
                    anz_img = classify_age(frame)
                    num_face, age_conf, gender_conf= confidence_mes(frame)
                    

                    kpi1_markdown.write(f"<h1 style='text-align: center;font-weight:normal; color:#bd4043; font-size:20px'>Number of Faces</h1>", unsafe_allow_html=True)
                    kpi1_text.write(f"<h1 style='text-align: center;font-weight:normal; color:white; font-size:18px'>{num_face}</h1>", unsafe_allow_html=True)
                    
                    kpi2_markdown.write(f"<h1 style='text-align: center;font-weight:normal; color:#bd4043; font-size:20px'>Gender Confidence</h1>", unsafe_allow_html=True)
                    kpi2_text.write(f"<h1 style='text-align: center;font-weight:normal; color:white; font-size:18px'>{gender_conf}</h1>", unsafe_allow_html=True)
                    
                    kpi3_markdown.write(f"<h1 style='text-align: center;font-weight:normal; color:#bd4043; font-size:20px'>Age Confidence</h1>", unsafe_allow_html=True)
                    kpi3_text.write(f"<h1 style='text-align: center;font-weight:normal; color:white; font-size:18px'>{age_conf}</h1>", unsafe_allow_html=True)


                   
                    out.write(anz_img)
            
                else:
                    break
                stframe.image(anz_img)
                
            out.release()
            vid.release()
    
            
            
    st.write("<footer style='text-align:center; color:gray; font-size:14px;'>Designed and Created by Nidhal Jegham | 2024 | Powered by Streamlit</footer>", unsafe_allow_html=True)
           
            
                            
                
                
                
    
    
    
    
    
    
    
    
    
if __name__=='__main__':
    main()
