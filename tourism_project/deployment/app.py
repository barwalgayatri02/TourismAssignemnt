import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib


api = HfApi(token=os.getenv("HF_TOKEN"))

# 🔹 Download and load the trained model
model_path = hf_hub_download(
    repo_id = "NaikGayatri/TourismAssignemnt",
    filename="best_tourism_project_model_v1.joblib" 
)
model = joblib.load(model_path)

# Streamlit UI for Customer Purchase Prediction
st.title("Customer Purchase Prediction App")
st.write("""
This application predicts whether a customer will purchase (`ProdTaken`) 
based on demographic and behavioral attributes.
Please fill the details below:
""")

# 🔹 User inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)
typeof_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
city_tier = st.selectbox("City Tier", [1, 2, 3])
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=60, value=15)
occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
gender = st.selectbox("Gender", ["Male", "Female"])
number_of_person_visiting = st.number_input("Number Of Person Visiting", min_value=1, max_value=10, value=2)
number_of_followups = st.number_input("Number Of Followups", min_value=0, max_value=10, value=2)
product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
preferred_property_star = st.selectbox("Preferred Property Star", [3, 4, 5])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
number_of_trips = st.number_input("Number Of Trips", min_value=0, max_value=20, value=5)
passport = st.selectbox("Passport", [0, 1])
pitch_satisfaction_score = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
own_car = st.selectbox("Own Car", [0, 1])
number_of_children_visiting = st.number_input("Number Of Children Visiting", min_value=0, max_value=5, value=0)
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=1000000, value=50000)

# 🔹 Assemble inputs into DataFrame
input_data = pd.DataFrame([{
    'Age': age,
    'TypeofContact': typeof_contact,
    'CityTier': city_tier,
    'DurationOfPitch': duration_of_pitch,
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': number_of_person_visiting,
    'NumberOfFollowups': number_of_followups,
    'ProductPitched': product_pitched,
    'PreferredPropertyStar': preferred_property_star,
    'MaritalStatus': marital_status,
    'NumberOfTrips': number_of_trips,
    'Passport': passport,
    'PitchSatisfactionScore': pitch_satisfaction_score,
    'OwnCar': own_car,
    'NumberOfChildrenVisiting': number_of_children_visiting,
    'Designation': designation,
    'MonthlyIncome': monthly_income
}])

# 🔹 Prediction
if st.button("Predict Purchase"):
    prediction = model.predict(input_data)[0]
    result = "✅ Customer Will Purchase (ProdTaken=1)" if prediction == 1 else "❌ Customer Will Not Purchase (ProdTaken=0)"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
