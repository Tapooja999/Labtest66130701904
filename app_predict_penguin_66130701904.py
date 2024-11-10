import streamlit as st
import joblib
import numpy as np

# โหลดโมเดล Decision Tree ที่บันทึกไว้
model = joblib.load('Decision_Tree_model.pkl')

# สร้าง dictionary เพื่อแปลงจากผลลัพธ์ตัวเลขเป็นชื่อสายพันธุ์
species_dict = {0: "Adelie", 1: "Chinstrap", 2: "Gentoo"}

# ฟังก์ชันทำนายประเภทของเพนกวิน
def predict_penguin(species_features):
    prediction = model.predict([species_features])
    return species_dict[prediction[0]]  # แปลงผลลัพธ์เป็นชื่อสายพันธุ์

# UI บน Streamlit
st.title("Penguin Species Prediction")
st.write("กรุณาใส่ข้อมูลของเพนกวินเพื่อทำนาย species")

# รับข้อมูล input จากผู้ใช้
bill_length = st.number_input("Bill Length (mm)")
bill_depth = st.number_input("Bill Depth (mm)")
flipper_length = st.number_input("Flipper Length (mm)")
body_mass = st.number_input("Body Mass (g)")

# แปลงข้อมูล input ให้เป็น array
input_features = np.array([bill_length, bill_depth, flipper_length, body_mass])

# ปุ่มทำนาย
if st.button("Predict"):
    species = predict_penguin(input_features)
    st.write(f"The predicted species is: {species}")
