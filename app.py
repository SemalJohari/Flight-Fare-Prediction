import pickle as pk
import streamlit as st
import numpy as np

@st.cache_resource
def load_model():
    return pk.load(open('model.pkl', 'rb'))

model = load_model()

airline_mapping = {'SpiceJet': 4, 'AirAsia': 0, 'Vistara': 5, 'GO_FIRST': 2, 'Indigo': 3, 'Air_India': 1}
source_city_mapping = {'Delhi': 2, 'Mumbai': 5, 'Bangalore': 0, 'Kolkata': 4, 'Hyderabad': 3, 'Chennai': 1}
departure_time_mapping = {'Evening': 2, 'Early_Morning': 1, 'Morning': 4, 'Afternoon': 0, 'Night': 5, 'Late_Night': 3}
stops_mapping = {'zero': 2, 'one': 1, 'two_or_more': 0}
arrival_time_mapping = {'Night': 5, 'Morning': 4, 'Early_Morning': 1, 'Afternoon': 0, 'Evening': 2, 'Late_Night': 3}
destination_city_mapping = {'Mumbai': 5, 'Bangalore': 0, 'Kolkata': 4, 'Hyderabad': 3, 'Chennai': 1, 'Delhi': 2}
class_mapping = {'Economy': 0, 'Business': 1}

introduction = """
### About the App

This machine learning application made using Numpy, Streamlit, Pandas, and Scikit-learn
predicts the fare of flights based on the following features: Airline, Source City,
Departure Time, Number of Stops, Arrival Time, Destination City, Class, Duration and the
number of days left.
Provide the inputs below to get the predicted flight fare.
"""

def display_instruction_window():
    with st.expander("💡 Information about the features", expanded=False):
        st.markdown("""
            #### Terminologies:
            1. Airline: The airline operating the flight.
            2. Source City: The city from which the flight originates.
            3. Departure Time: Time of departure (categorized into slots).
            4. Number of Stops: Stops between source and destination.
            5. Arrival Time: Time of arrival (categorized into slots).
            6. Destination City: The city where the flight lands.
            7. Class: The class of the flight (Economy or Business).
            8. Duration: Flight duration in hours (ranges from 2 to 4 hours).
            9. Days Left: Number of days left before the departure date.
            """)

st.title("Flight Fare Prediction")
st.markdown(introduction)

col1, col2, col3 = st.columns(3)

with col1:
    airline = st.selectbox('Select Airline', list(airline_mapping.keys()))
    departure_time = st.selectbox('Select Departure Time', list(departure_time_mapping.keys()))
    stops = st.selectbox('Select Stops', list(stops_mapping.keys()))

with col2:
    source_city = st.selectbox('Select Source City', list(source_city_mapping.keys()))
    arrival_time = st.selectbox('Select Arrival Time', list(arrival_time_mapping.keys()))
    destination_city = st.selectbox('Select Destination City', list(destination_city_mapping.keys()))

with col3:
    flight_class = st.selectbox('Select Class', list(class_mapping.keys()))
    duration = st.number_input('Duration (in hours)', min_value=1.00, max_value=4.00, format="%.2f")
    days_left = st.slider('Days Left Before Departure', min_value=1, max_value=49, step=1)

with st.sidebar:
        st.title("More Information")
        display_instruction_window()

if st.button('Predict Fare'):

    features = [
    airline_mapping[airline],
    source_city_mapping[source_city],
    departure_time_mapping[departure_time],
    stops_mapping[stops],
    arrival_time_mapping[arrival_time],
    destination_city_mapping[destination_city],
    class_mapping[flight_class],
    duration,
    days_left,
    ]

    features_array = np.array(features).reshape(1, -1)

    predicted_fare = model.predict(features_array)[0]

    st.success(f"### The predicted flight fare is: ₹{predicted_fare:.2f}")
