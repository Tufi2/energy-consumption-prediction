# frontend/app.py
import streamlit as st
from pages import home, predictions, analysis
import base64
import sys
import os

# Add src to path for importing backend modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def load_css():
    with open('frontend/static/css/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def main():
    st.set_page_config(
        page_title="Energy Consumption Predictor",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load custom CSS
    load_css()
    
    # Add background image
    # add_bg_from_local('frontend/static/images/background.png')
    
    # Custom sidebar
    with st.sidebar:
        st.markdown("""
            <div class="sidebar-header">
                ⚡ Energy Predictor
            </div>
        """, unsafe_allow_html=True)
        
        selected = st.selectbox(
            "",
            ["Home", "Predictions", "Analysis"],
            format_func=lambda x: f"📍 {x}"
        )
    
    # Page routing with animations
    if selected == "Home":
        home.show()
    elif selected == "Predictions":
        predictions.show()
    elif selected == "Analysis":
        analysis.show()

if __name__ == "__main__":
    main()