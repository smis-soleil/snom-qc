"""
This pages shows a map QC report for the uploaded document.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import utils
from utils import SessionStateSingleton

# Set up sidebar (navigation and uploader to change file)
utils.setup_page_with_redirect(allowed_file_types=['axz', 'axd'])

st.title('Map QC report')
st.write(
    '''
    This page shows a map QC report for the uploaded document. The report includes a visualisation of all maps at a given timestamp, as well as a download link for a PDF version of the report. We also check for common issues such as high fluctutions in the deflection or IR phase signal, PLL saturation, and data processing errors. This is based on simple heuristics and should not be considered a definitive QC report.

    Note that heightmaps and deflection maps are processed (linear fit and setpoint subtraction, respectively) before plotting the image, yet the trace plots show the original data.
    
    '''
)

file_hash = SessionStateSingleton().get_file_hash()
doc_uploaded = SessionStateSingleton().get_anasys_doc()
map_df = SessionStateSingleton().get_cached_heightmap_groups()

# Main content
if len(doc_uploaded.HeightMaps) == 0:
    st.error('No maps found. Choose another file')
    st.stop()

ncols = max(map_df['Map Labels'].apply(len))

dowload_placeholder = st.empty()

if 'allmaps' in st.session_state.cached_image_qa:
    buttondata = st.session_state.cached_image_qa['allmaps']
    buttonddisabled = False
    buttontext = 'Download QC report as PDF'
else:
    buttondata = b''
    buttonddisabled = True
    buttontext = 'Generating PDF report...'

dowload_placeholder.download_button(
    label=buttontext,
    data=buttondata,
    file_name=f'{SessionStateSingleton().get_file_name().replace('.', '_')}_mapqc.pdf',
    type='primary',
    icon=':material/download:',
    disabled=buttonddisabled,
    key='disabled-button'
)

map_warnings = {
    ts: utils.collect_map_qc_warnings(doc_uploaded, map_df.loc[ts, 'Map Labels'])
    for ts in map_df.index
}

for ts in map_df.index:
    st.pyplot(SessionStateSingleton().get_cached_image_qa(ts, ncols))
    if len(map_warnings[ts]) > 0:
        st.warning(icon=':material/warning:', body='The following potential issues were detected:\n\n* ' + '\n\n* '.join(map_warnings[ts]))

if all(len(map_warnings[ts]) == 0 for ts in map_df.index):
    st.success('No potential issues were detected.', icon=':material/check_circle:')

# Plot QC maps
if 'allmaps' not in st.session_state.cached_image_qa:
    dowload_placeholder.download_button(
        label='Download QC report as PDF',
        data=SessionStateSingleton().get_cached_image_qa(allmaps=True, ncols=ncols),
        file_name=f'{SessionStateSingleton().get_file_name().replace('.', '_')}_mapqc.pdf',
        type='primary',
        icon=':material/download:',
    )