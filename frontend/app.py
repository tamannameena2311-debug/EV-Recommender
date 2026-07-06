import streamlit as st
import requests
import pandas as pd

import os

# Constants
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="EV Car Recommender", layout="wide")

st.title("Electric Vehicle (EV) Recommendation System")
st.markdown("Find the best electric car in India based on your personal preferences.")

with st.sidebar:
    st.header("Your Preferences")
    
    budget = st.slider("Budget (in Lakhs)", min_value=5.0, max_value=80.0, value=15.0, step=0.5)
    minimum_range = st.slider("Minimum Range (km)", min_value=100, max_value=600, value=250, step=10)
    daily_travel = st.number_input("Daily Travel (km)", min_value=1, max_value=500, value=40)
    
    city = st.text_input("City", value="Delhi")
    state = st.text_input("State", value="Delhi")
    
    use_case = st.selectbox(
        "Primary Use Case", 
        ["daily_city_commute", "family_use", "highway_travel", "office_commute", "budget_friendly", "performance", "premium"]
    )
    
    preferred_body_type = st.selectbox(
        "Preferred Body Type",
        ["Any", "SUV", "Hatchback", "Sedan", "Crossover", "Compact SUV", "Mid-size SUV"]
    )
    
    family_size = st.number_input("Family Size (Seating Capacity)", min_value=2, max_value=8, value=4)
    
    home_charging = st.checkbox("Home Charging Available?", value=True)
    fast_charging = st.checkbox("Need Fast Charging?", value=True)
    
    brand_preference = st.selectbox(
        "Brand Preference",
        ["Any", "Tata", "MG", "Mahindra", "Hyundai", "Kia", "BYD", "Citroen"]
    )
    
    priority = st.selectbox(
        "What is your highest priority?",
        ["balanced", "lowest_price", "maximum_range", "fast_charging", "family_comfort", "safety", "performance"]
    )
    
    submit_button = st.button("Find My EV", type="primary")

if submit_button:
    with st.spinner("Finding the best EVs for you..."):
        payload = {
            "budget_lakh": budget,
            "minimum_range_km": minimum_range,
            "daily_travel_km": daily_travel,
            "city": city,
            "state": state,
            "use_case": use_case,
            "preferred_body_type": preferred_body_type,
            "family_size": family_size,
            "home_charging_available": home_charging,
            "fast_charging_needed": fast_charging,
            "brand_preference": brand_preference,
            "priority": priority
        }
        
        try:
            response = requests.post(f"{API_URL}/recommend", json=payload)
            response.raise_for_status()
            data = response.json()
            
            recommendations = data.get("recommendations", [])
            
            if not recommendations:
                st.warning("No cars found matching your strict criteria. Try relaxing your budget or family size.")
            else:
                st.success(f"Found {len(recommendations)} recommended EVs for you!")
                
                # Display recommendations
                for rec in recommendations:
                    with st.expander(f"#{rec['rank']} - {rec['car_name']} ({rec['match_percentage']}% Match)", expanded=(rec['rank']==1)):
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.markdown(f"**Reason:** {rec['reason']}")
                            st.markdown(f"**Drawbacks:** {rec['drawbacks']}")
                        with col2:
                            st.metric("Price (On-Road)", f"₹{rec['price_lakh']} L")
                            st.metric("Claimed Range", f"{rec['claimed_range_km']} km")
                            st.metric("Battery", f"{rec['battery_capacity_kwh']} kWh")
                            
                st.subheader("Comparison Table")
                df_recs = pd.DataFrame(recommendations)
                # Select important columns to display
                df_display = df_recs[['rank', 'car_name', 'brand', 'price_lakh', 'claimed_range_km', 'match_percentage']]
                df_display.columns = ['Rank', 'Car', 'Brand', 'Price (Lakh)', 'Range (km)', 'Match Score (%)']
                st.dataframe(df_display, use_container_width=True)
                
                # Download CSV
                csv = df_recs.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Recommendations as CSV",
                    data=csv,
                    file_name='ev_recommendations.csv',
                    mime='text/csv',
                )
                
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to backend API. Please make sure the FastAPI server is running. ({e})")
