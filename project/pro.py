import streamlit as st
import pandas as pd
from io import StringIO
import tensorflow as tf
import numpy as np
import io
import pathlib
from PIL import Image
import cv2
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from array import array
import joblib
import streamlit as st







# Define the HTML content and CSS style for the homepage
st.set_page_config(page_title='Diagnostic',layout="wide",page_icon="resou/large.png")  # Set the page title


# col_1, col_2, col_3 ,col_4,col_5= st.columns(5)
# with col_1:
#     pass
# with col_2:
#  pass
# with col_3:
#      st.image("resou/large.png",width=200)
# with col_4:

#  pass
# with col_5:
#     pass
st.markdown('''
    <a >
        <img src="https://i.imgur.com/ipAltMw.png",width=200" width="130" height="130"/>
  
    </a>
 
    ''',
    unsafe_allow_html=True
)

 
homepage_html = """
    <html>
    <head>
        <style>
            body {
                font-family: serif;;
                text-align: center;

                border-style:double;
                border-width:5px;
                border-color:#FFFAF0;
                border-radius:3px;
                
            }
            
            
            h1 {
                color: #0072B2;
                font-size: 52px;
                margin-top: 100px;
               
                    }
            p {
                color: #24DAD7;
                font-size: 24px;
                margin-top: 20px;
            }
        </style>
    </head>
    <body>
        <h1>Diagnostic</h1>
        <p>Because each and every life matter!</p>
        <!-- Add your custom HTML content here -->
        <div class="column">
        
    </div>
       
        
       
     
    </body>
    </html>
"""

# Define the CSS style for the page
page_css = """

	.image1{
             position: absolute;
             bottom: 16px;
             right: 14px;  			 
}
	
    body {
        padding: 0;
        margin: 0;
    }
    .stApp {
        padding-top: 0;
        padding-left: 0;
        padding-right: 0;
        padding-bottom: 0;
    }
"""



tab_1, tab1, tab2, tab3 , tab4 = st.tabs(["Home",'Brain', "Diabetes", "HeartAttack",'Other'])
with tab_1:

 st.markdown(homepage_html, unsafe_allow_html=True)  # Render the HTML content using st.markdown
 st.markdown(f'<style>{page_css}</style>', unsafe_allow_html=True) 

    
with tab1:


    uploaded_file = st.file_uploader("Choose a file",type=["png", "jpg", "jpeg"])

    loading=tf.keras.models.load_model("brain_model.h5")


    data_dir=pathlib.Path("Tumor_MRI/brain_tumor_dataset/")

    class_name=class_name=np.array(sorted([item.name for item in data_dir.glob("*")]))

    def load_prep_img(filename,img_shape=224):

      # read in the image
      img=tf.io.read_file(filename)
      # Decode the read file into a tensor
      img=tf.image.decode_image(img)
      # resize the image
      img=tf.image.resize(img,size=[img_shape,img_shape])
      # Rescale the image (get all values between 0 and 1)
      img=img/225

      bytesImg = io.BytesIO(filename)
      img = Image.open(bytesImg)   
      

      return img

    def pred_and_plot(filename,class_name=class_name):
        """
        Imports an image locate at filename ,make a prediction with model
        and plot the images with the predicted class as title
        """
      #  import the target image and preprocess it
        img=load_prep_img(filename)
        pred=loading.predict(tf.expand_dims(img,axis=0))
        pred_class=class_name[int(tf.round(pred))]
        st.write(pred_class)


      #  pred=loading.predict(tf.expand_dims(img,axis=0))
      #       pred_class=class_name[int(tf.round(pred))]
    def saveImage(byteImage):
        bytesImg = io.BytesIO(byteImage)
        imgFile = Image.open(bytesImg)   
      
        return imgFile
    if uploaded_file is not None:

        file=uploaded_file.read()
        file=tf.image.decode_image(file)
        file=tf.image.resize(file,size=[224,224])
        # path = saveImage(file)
        # st.image(path)
        # file= load_prep_img(file)
        pred=loading.predict(tf.expand_dims(file,axis=0))
        pred_class=class_name[int(tf.round(pred))]
        st.title(pred_class)
        with st.sidebar:
      
          st.image(uploaded_file)
  
        col1, col2, col3 = st.columns(3)

        with col1:
          
          pass

        with col2:
         st.image(uploaded_file)


        with col3:
          pass
        st.success("Done!")

## Diabites

with tab2:
          Data=pd.read_csv('diabetes (1).csv')

          Data.head()
          data=Data.drop('Outcome',axis=1)
          outc=Data["Outcome"]

          ct=make_column_transformer(
                  (MinMaxScaler(),['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']),
              )
          outc=Data["Outcome"]

          X=Data.drop("Outcome",axis=1)

          y=Data["Outcome"]
          X_train,X_test,Y_train,Y_test=train_test_split(data,outc,test_size=0.2,random_state=42)
          ct.fit(X_train)

          col1, col2 = st.columns(2)
          col3, col4 =st.columns(2)
          col5, col6 =st.columns(2)
          col7, col8 =st.columns(2)

          with col1:
            #  number = st.number_input('Glucose')
            Pregnancies=  st.slider('How many pregnancies?', 0, 10, 0)

          with col2:
             Glucose = st.number_input('Glucose')
          
          with col3:
             BloodPressure=st.number_input('BloodPressur')
          with col4:
             SkinThickness=st.number_input('SkinThickness')
          with col5:
            Insulin=st.number_input('Insulin')
          with col6:
            BMI=st.number_input('BMI')
          with col7:
            DiabetesPedigreeFunction=st.number_input('DiabetesPedigreeFunction')
          with col8:
            Age=st.slider('Age', 0, 100, 0)
          # st.write(Pregnancies+Glucose+BloodPressure+SkinThickness+Insulin+BMI+DiabetesPedigreeFunction+Age)
          test={'Pregnancies':[Pregnancies],'Glucose':[Glucose],'BloodPressure':[BloodPressure],'SkinThickness':[SkinThickness],'Insulin':[Insulin],'BMI':[BMI],'DiabetesPedigreeFunction':[DiabetesPedigreeFunction],'Age':[Age]}
          df=pd.DataFrame(test)
          df=ct.transform(df)
          loading2=tf.keras.models.load_model("diabetes.h5")
          x=loading2.predict(df)
          btn1=st.button('Show Prediction')
          if btn1:
                if tf.math.round(x)==1:
                      st.title('positive!!')
                      st.warning('This AI model has accuracy of 80 percent though seek expert advice The developer take no liability of any false prediction', icon="⚠️")
                    
                else:
                      st.title('Negative!')
                      # st.warning('This AI model has accuracy of 80 percent though seek expert advice The developer take no liability of any false prediction', icon="")
                      st.markdown(":red['This AI model has accuracy of 80 percent though seek expert advice The developer take no liability of any false prediction'], **:blue[⚠️]** ")     
# Heart ❤️            
with tab3:
  

          col1, col2 = st.columns(2)
          col3, col4 =st.columns(2)
          col5, col6 =st.columns(2)
        

          with col1:
            #  number = st.number_input('Glucose')
            Age=  st.slider('Your Age', 0, 100, 0)
          with col2:
             CP = st.number_input('CP')
          with col3:
            trestbps=st.number_input('trestbps')
          with col4:
             Cholesterol =st.number_input('Cholesterol ')
          with col5:
            thalachh=st.number_input('Maximum heart rate achieved')
          with col6:
            oldpeak=st.number_input('Previous peak')
       
          # st.write(Pregnancies+Glucose+BloodPressure+SkinThickness+Insulin+BMI+DiabetesPedigreeFunction+Age)
          test={'Age':[Age],'CP':[CP],'trestbps':[trestbps],'Cholesterol':[Cholesterol],'thalachh':[thalachh],'oldpeak':[oldpeak]}
          df=pd.DataFrame(test)
          # X = np.asarray(test).astype(np.float32)

       
          loading3=tf.keras.models.load_model("Heart_attack.h5")
          x= loading3.predict(df)
          col_1, col_2,col_3 = st.columns(3)
          # with col_1:
          #   pass

          # with col_2:
          b1=st.button('ShowPrediction',key=1)
          if b1:
              if tf.math.round(x)==1:
                        st.title('positive!!')
                        st.warning('This AI model has accuracy of 80 percent though seek expert advice The developer take no liability of any false prediction', icon="⚠️")
                      
              else:
                        st.title('Negative!')
                        st.warning('This AI model has accuracy of 80 percent though seek expert advice The developer take no liability of any false prediction', icon="⚠️")
                      

with tab4:
  msg="1 if the symptom is present, 0 if it isn't"
  st.write(msg)
  # excessive_hunger=st.number_input("excessive hunger",min_value=0,max_value=1)
  lst1=[
"itching",
"skin_rash",
'nodal_skin_eruptions',
'continuous_sneezing',
"shivering",
"chills",
"joint_pain",
"stomach_pain",
"acidity",
"ulcers_on_tongue",
"muscle_wasting",
"vomiting",
"burning_micturition",
"spotting_ urination",
"fatigue",
"weight_gain",
"anxiety",
"cold_hands_and_feets",
"mood_swings",
"weight_loss",
"restlessness",
"lethargy",
"patches_in_throat",
"irregular_sugar_level",
"cough",
"high_fever",
"sunken_eyes",
"breathlessness",
"sweating",
"dehydration",
"indigestion",
"headache",
"yellowish_skin",
"dark_urine",
"nausea",
"loss_of_appetite",
"pain_behind_the_eyes",
"back_pain",
"constipation",
"abdominal_pain",
"diarrhoea",
"mild_fever",
"yellow_urine",
"yellowing_of_eyes",
"acute_liver_failure",
"fluid_overload",
"swelling_of_stomach",
"swelled_lymph_nodes",
"malaise",
"blurred_and_distorted_vision",
"phlegm",
"throat_irritation",
"redness_of_eyes",
"sinus_pressure",
"runny_nose",
"congestion",
"chest_pain",
"weakness_in_limbs",
"fast_heart_rate",
"pain_during_bowel_movements",
"pain_in_anal_region",
"bloody_stool",
"irritation_in_anus",
"neck_pain",
"dizziness",
"cramps",
"bruising",
"obesity",
"swollen_legs",
"swollen_blood_vessels",
"puffy_face_and_eyes",
"enlarged_thyroid",
"brittle_nails",
"swollen_extremeties",
"excessive_hunger",
"extra_marital_contacts",
"drying_and_tingling_lips",
"slurred_speech",
"knee_pain",
"hip_joint_pain",
"muscle_weakness",
"stiff_neck",
"swelling_joints",
"movement_stiffness",
"spinning_movements",
"loss_of_balance",
"unsteadiness",
"weakness_of_one_body_side",
"loss_of_smell",
"bladder_discomfort",
"foul_smell_of urine",
"continuous_feel_of_urine",
"passage_of_gases",
"internal_itching",
"toxic_look_(typhos)",
"depression",
"irritability",
"muscle_pain",
"altered_sensorium",
"red_spots_over_body",
"belly_pain",
"abnormal_menstruation",
"dischromic _patches",
  "watering_from_eyes",
  "increased_appetite",
  "polyuria",
  "family_history",
  "mucoid_sputum",
  "rusty_sputum",
  "lack_of_concentration",
  "visual_disturbances",
  "receiving_blood_transfusion",
  "receiving_unsterile_injections",
  "coma",
  "stomach_bleeding",
  "distention_of_abdomen",
  "history_of_alcohol_consumption",
  "fluid_overload.1",
  "blood_in_sputum",
  "prominent_veins_on_calf",
  "palpitations",
  "painful_walking",
  "pus_filled_pimples",
  "blackheads",
  "scurring",
  "skin_peeling",
  "silver_like_dusting",
  "small_dents_in_nails",
  "inflammatory_nails",
  "blister",
  "red_sore_around_nose",
  "yellow_crust_ooze",
  ]
  mapp={
     0:'(vertigo) Paroymsal', 1:'AIDS' , 2:'Acne',  3:'Alcoholic hepatitis' , 4:'Allergy',  5:'Arthritis', 6:'Bronchial Asthma',7:'Cervical spondylosis',8:'Chicken pox',9:'Chronic cholestasis', 10:'Common Cold',11:'Dengue',12:'Diabetes',13:"Dimorphic hemmorhoids(piles)",14:'Drug Reaction', 15:'Fungal infection',16:'GERD',17:'Gastroenteritis', 18:'Heart attack',19:'Hepatitis B',20:'Hepatitis C', 21:"Hepatitis D", 22:"Hepatitis E", 23:'Hypertension',
    24:'Hyperthyroidism', 25:'Hypoglycemia', 26:"Hypothyroidism", 27:'Impetigo', 28:"Jaundice", 29:"Malaria", 30:'Migraine', 31:'Osteoarthristis', 32:"Paralysis (brain hemorrhage)", 33:"Peptic ulcer diseae", 34:'Pneumonia', 35:"Psoriasis", 36:"Tuberculosis", 37:"Typhoid" ,38:"Urinary tract infection", 39:"Varicose veins",40:"hepatitis A"
 }
  lst2=[]
  print("leng",len(lst1))
  for i in range(len(lst1)):
        name=lst1[i]
        global arry
        Symp=st.number_input(label=name,min_value=0,max_value=1)
        lst2.append(Symp)
  print(lst2)
  print(len(lst2))

  loaded_model = joblib.load("finalized_model.sav")
  arr = np.array(lst2)
  arr=arr.reshape(1, -1)
  diseas=loaded_model.predict(arr)
  b2=st.button('ShowPrediction',key=2)
  if b2:
              st.title(mapp[diseas[0]])
              st.warning('This AI model has fairly high accuracy though seek expert advice The developer take no liability of any false prediction', icon="⚠️")


hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
st.markdown('''
    <a href="mailto:henilsinhrajraj@gmail.com">
        <img src="https://cdn.iconscout.com/icon/free/png-512/free-gmail-3521453-2944897.png?f=avif&w=512" width="50" height="50"/>
     <a href="mailto:henilsinhrajraj@gmail.com">
        <img src="https://cdn.iconscout.com/icon/premium/png-512-thumb/review-2055819-1734040.png?f=avif&w=512" width="50" height="50"/>
    </a>
    </a>
 
    ''',
    unsafe_allow_html=True
)

