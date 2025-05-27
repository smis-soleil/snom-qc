import streamlit as st
# from utils import SessionStateSingleton
import utils

utils.setup_page()
st.page_link(
    'app.py',
    label='Back to the home page',
    icon=':material/arrow_back:',
)
st.video('source/demo.mp4')