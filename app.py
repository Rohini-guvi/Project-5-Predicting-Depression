import streamlit as st

current_page=st.navigation([st.Page("Home.py"),
                            st.Page("Predict.py", title="Mental Health Prediction")
                            ])

current_page.run()