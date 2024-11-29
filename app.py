import streamlit as st
from Data_Preprocessing_Interval import *
from save_load_model import *

# Load the saved model
model=load_model("Random_Forest_Regressor_model_Time_Interval.pkl")
#print(model.coef_)

# Title and description
st.title("Rented Bike Count Prediction")
st.write("""
Provide the required features, and this app will predict the rented bike count.
Ensure you input all features correctly as per preprocessing steps.
""")

# Input fields for features
with st.form("input_form"):
    st.header("Input Features")

    # Date-related features
    # Date-related features
    date = st.date_input("Date")
    Time_Interval = st.selectbox("Time Interval?", ["Morning", "Afternoon","Evening","Night"])
    temperature = st.number_input("Temperature (in °C)", min_value=-10.0, max_value=50.0, step=0.1)
    wind_speed = st.number_input("Wind Speed (m/s)", min_value=0.0, max_value=20.0, step=0.1)
    solar_radiation = st.number_input("Solar Radiation (MJ/m2)", min_value=0.0, max_value=3.0, step=0.1)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=50.0, step=0.1)

    # Functioning Day
    functioning_day = st.selectbox("Is it a Functioning Day?", ["Yes", "No"])
    functioning_day = 1 if functioning_day == "Yes" else 0

    # Holiday
    holiday = st.selectbox("Is it a Holiday?", ["No Holiday", "Holiday"])
    holiday = 0 if holiday == "No Holiday" else 1

    # Seasons
    season = st.selectbox("Season", ["Spring", "Summer", "Autumn", "Winter"])

    # Submit button
    submitted = st.form_submit_button("Predict")

# Handle predictions
if submitted:
    # Preprocess the input data
    input_data,scaling_factor = pre_process(Time_Interval, temperature, rainfall, solar_radiation, wind_speed, holiday, functioning_day, season,date)
    # Predict using the trained model
    prediction = model.predict(input_data)
    print(prediction)
    #rented_bike_count_exp=np.expm1(prediction[0])
    rented_bike_count = np.square(prediction[0])
    print(rented_bike_count)
    print(scaling_factor)  # Reverse the sqrt transformation
    #rented_bike_count = rented_bike_count*scaling_factor # Reverse the sqrt transformation

    # Display the result
    st.success(f"Predicted Rented Bike Count: {rented_bike_count:.2f}")
