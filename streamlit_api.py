import pickle
import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

st.markdown("## Bundesministerium für Gesundheit")



def load_pickles(model_pickle_path, label_encoder_pickle_path):
    """
    Loading the saved pickles with model and label encoder
    """
    model_pickle_opener = open(model_pickle_path, "rb")
    model = pickle.load(model_pickle_opener)

    label_encoder_pickle_opener = open(label_encoder_pickle_path, "rb")
    label_encoder_dict = pickle.load(label_encoder_pickle_opener)

    return model, label_encoder_dict

def pre_process_data(df, label_encoder_dict):
    """
    Apply pre-processing steps from before to new data
    """
    for col in df.columns:
        if col in list(label_encoder_dict.keys()):
            # accessing the column's label encoder via
            # column name as key
            column_le = label_encoder_dict[col]
            # applying fitted label encoder to the data
            df.loc[:, col] = column_le.transform(df.loc[:, col])
        else:
            continue
    return df

def make_predictions(processed_df, model):
    prediction = model.predict(processed_df)
    return prediction

def generate_predictions(test_df):
    model_pickle_path = "heart_attack_prediction_model.pkl"
    label_encoder_pickle_path = "heart_attack_prediction_label_encoders.pkl"

    model, label_encoder_dict = load_pickles(model_pickle_path,
                                             label_encoder_pickle_path)

    processed_df = pre_process_data(test_df, label_encoder_dict)
    prediction = make_predictions(processed_df, model)

    return prediction

if __name__ == "__main__":
    st.title("Heart Attack Predictor")
    st.subheader("Enter data about the cusomter you would like to predict:")

    # making customer data inputs
    age = st.slider("Select patient's age:",
                          min_value=0, max_value=100, value=50)
    gender = st.selectbox("What's the patiente's gender?:",
                          ['Man', "Woman"])
    
    if gender == "Woman":
        gender = 0
    else:
        gender = 1
    cp = st.selectbox("Does the patiente have any chest pain?:",
                          ['typical angina', "atypical angina", "non-anginal pain", "asymptomatic"])

    if cp == "typical angina":
        cp = 0
    else:
        if cp == "atypical angina":
            cp = 1
        else:
            if cp == "non-anginal pain":
                cp = 2
            else:
                cp = 3

    trtbps = st.slider("resting blood pressure?:",
                        min_value=94, max_value=200, value=120)
    chol = st.slider("cholestoral in blood  ?:",
                        min_value=126, max_value=564, value=140)

    fbs = st.selectbox("fasting blood sugar?:",
                                        ["Yes", "No"])
    if fbs == "Yes":
        fbs = 1
    else:
        fbs = 0

    rest_ecgp = st.selectbox("resting electrocardiographic results?:",
                          ['normal', "having ST-T wave abnormality", "showing probable or definite left ventricular hypertrophy by Estes criteria"])

    if rest_ecgp == "normal":
        rest_ecgp = 0
    else:
        if rest_ecgp == "having ST-T wave abnormality":
            rest_ecgp = 1
        else:
            rest_ecgp = 2
    

    thalach  = st.slider("maximum heart rate achieved?:",
                         min_value=71, max_value=200, value=72)

    exng = st.selectbox("exercise induced angina?:",
                                        ["Yes", "No"])
    if exng == "Yes":
        exng = 1
    else:
        exng = 0

    caa = st.selectbox("Number of major vessels?",
    ["0", "1", "2", "3"])

    if caa == "0":
        caa = 0
    else:
        if caa == "1":
            caa = 1
        else:
            if caa == "2":
                caa = 2
            else:
                if caa == "3":
                    caa = 3
            

    o2saturation = st.slider("o2saturation?",
    min_value=96.5, max_value=98.6, value=96.5)
    
    input_dict = {"age": age,
                  "gender": gender,
                  "cp": cp,
                  "trtbps": trtbps,
                  "chol": chol,
                  "fbs": fbs,
                  "rest_ecgp": rest_ecgp,
                  "thalach": thalach,
                  "exng": exng,
                  "caa": caa,
                  "o2saturation": o2saturation}

    input_data = pd.DataFrame([input_dict])

    if st.button("Predict Heart attack"):
        pred = generate_predictions(input_data)
        if bool(pred):
            st.error("Customer won't have a heart attack")
            st.snow()
        else:
            st.success("Customer may have a heart attack")

