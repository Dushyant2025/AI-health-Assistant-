
from newfea import *  
import streamlit as st
from chattt import *
# Custom CSS to make buttons the same size
st.markdown(
    """
    <style>
    /* Background and text color */
    body {
        background-color: #fff5e6;
        color: #ff8c00;
    }

    /* Button styling */
    .stButton>button {
        width: 100%;
        height: 100px;
        font-size: 38px;
        font-weight: bold;
        color: white;
        background-color: #ff8c00;
        border-radius: 10px;
        border: none;
        transition: 0.3s;
    }

    .stButton>button:hover {
        background-color: #e67e22;
        transform: scale(1.05);
        
    }

    
    }
    </style>
    """,
    unsafe_allow_html=True
)


if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Function to handle page navigation
def navigate_to(page):
    st.session_state.page = page

# Function to show home page
def home():
    st.header("🤖 DoctorWALA")
    st.title("Welcome to AI-Health-Assistant")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.button("🩺 Symptoms Analyzer", key='disease_prediction', on_click=navigate_to, args=('disease',))

    with col2:
        st.button("💡 Health Analyzer", key='health_analyzer', on_click=navigate_to, args=('analyzer',))


# Function to show disease prediction page
def disease_prediction():
    newfea()  # This will render the content from newfea.py
    st.button("🔙 Back to Home", on_click=navigate_to, args=('home',))

# Function to show health analyzer page
def health_analyzer():
    st.header("💡 Health Analyzer")
    analyse()
    st.button("🔙 Back to Home", on_click=navigate_to, args=('home',))

if st.session_state.page == 'home':
    home()
elif st.session_state.page == 'disease':
    disease_prediction()
elif st.session_state.page == 'analyzer':
    health_analyzer()
