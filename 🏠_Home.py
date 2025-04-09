import streamlit as st

def Main_App():
    st.set_page_config(
        page_title="Hazard Detector",
        layout="wide",
        page_icon="⛑️",
        initial_sidebar_state="expanded"
    )
    
    
    st.sidebar.title("Navigation")
    st.sidebar.success("Select a detection mode above")
    
    st.title("Safety Hazard Detection Application")
    st.write("""
    Welcome to the Safety Hazard Detection System. 
    Use the navigation in the sidebar to switch between detection modes.
    """)

if __name__ == "__main__":
    Main_App()