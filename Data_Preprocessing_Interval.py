import streamlit as st
import pandas as pd
import numpy as np
import pickle
from save_load_model import *
# Define the preprocessing function
def pre_process(Time_Interval, temperature, rainfall, solar_radiation, wind_speed, holiday, functioning_day, season,date):
    # Extract month and day information from date
    month = pd.to_datetime(date).month
    day_name = pd.to_datetime(date).day_name()

    # Determine if the day is a weekend
    is_weekend = 1 if day_name in ["Saturday", "Sunday"] else 0

    # Map time intervals
    if Time_Interval=="Morning":
        time_interval = 0  # Morning
    elif Time_Interval=="Afternoon":
        time_interval = 1  # Afternoon
    elif Time_Interval=="Evening":
        time_interval = 2  # Evening
    elif Time_Interval=="Night":
        time_interval = 3  # Night

    # Map seasons to dummy variables
    seasons_spring = 1 if season == "Spring" else 0
    seasons_summer = 1 if season == "Summer" else 0
    seasons_winter = 1 if season == "Winter" else 0

    # Create a DataFrame with the preprocessed data
    preprocessed_data = pd.DataFrame({
        "Time_Interval": [time_interval],
        "Temperature": [temperature],
        "Wind_speed": [wind_speed],
        "Solar_Radiation": [solar_radiation],
        "Rainfall": [rainfall],
        "Functioning_Day": [functioning_day],
        "Month":[month],
        "Weekdays_or_weekend": [is_weekend],
        "Holiday": [holiday],
        "Seasons_Spring": [seasons_spring],
        "Seasons_Summer": [seasons_summer],
        "Seasons_Winter": [seasons_winter]
    })

    print(preprocessed_data)

    # Apply MinMax scaling (ensure the scaler matches training-time preprocessing)
    scaler = load_scaler("scaler.pkl")
    scaled_data = scaler.transform(preprocessed_data)
    scaling_factor = scaler.scale_ 
    return scaled_data,scaling_factor 
