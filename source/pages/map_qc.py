"""
This pages shows a map QC report for the uploaded document.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from utils import setup_page_with_redirect, collect_map_qc_warnings, SessionState
import plotting  # pylint: disable=E0401

# Set up sidebar (navigation and uploader to change file)
setup_page_with_redirect(allowed_file_types=['axz', 'axd'])

st.title('Map QC report')
st.write(
    '''
    This page shows a map QC report for the uploaded document. The report includes a visualisation of all maps at a given timestamp, as well as a download link for a PDF version of the report. We also check for common issues such as high fluctutions in the deflection or IR phase signal, PLL saturation, and data processing errors. This is based on simple heuristics and should not be considered a definitive QC report.

    Note that heightmaps and deflection maps are processed (linear fit and setpoint subtraction, respectively) before plotting the image, yet the trace plots show the original data.
    
    '''
)

# Load map properties from uploaded file
@st.cache_resource
def list_heightmap_groups(file_hash):
    """
    Function to load map properties from uploaded file
    """

    doc = SessionState().get_anasys_doc()
    map_dict = {m.Label: m.TimeStamp for m in doc.HeightMaps.values()}
    df = pd.DataFrame([
        {
            'ts': ts,
            'Timestamp': pd.to_datetime(ts).strftime('%Y-%m-%d %H:%M:%S'),
            'Map Labels': set(m for m, v in map_dict.items() if v == ts)
        }
        for ts in set(map_dict.values())
    ]).set_index('ts').sort_index()

    return doc, df

file_hash = SessionState().get_file_hash()
doc_uploaded, map_df = list_heightmap_groups(file_hash)

# Main content
if len(doc_uploaded.HeightMaps) == 0:
    st.error('No maps found. Choose another file')
    st.stop()

ncols = max(map_df['Map Labels'].apply(len))

dowload_placeholder = st.empty()
dowload_placeholder.download_button(
    label='Generating PDF report...',
    data=b'',
    file_name=f'{SessionState().get_file_name()}_mapqc.pdf',
    type='primary',
    icon=':material/download:',
    disabled=True,
    key='disabled-button'
)

map_warnings = {
    ts: collect_map_qc_warnings(doc_uploaded, map_df.loc[ts, 'Map Labels'])
    for ts in map_df.index
}

for ts in map_df.index:
    st.pyplot(plotting.plot_maps_qc_at_timestamp(file_hash, ts, ncols=ncols))
    if len(map_warnings[ts]) > 0:
        st.warning(icon=':material/warning:', body='The following potential issues were detected:\n\n* ' + '\n\n* '.join(map_warnings[ts]))

if all(len(map_warnings[ts]) == 0 for ts in map_df.index):
    st.success('No potential issues were detected.', icon=':material/check_circle:')

# Plot QC maps

# dowload_placeholder.empty()
dowload_placeholder.download_button(
    label='Download QC report as PDF',
    data=plotting.generate_mapqc_pdf(file_hash, ncols=ncols),
    file_name=f'{SessionState().get_file_name()}_mapqc.pdf',
    type='primary',
    icon=':material/download:',
)

# with st.spinner('Plotting data...'):
#     st.pyplot(plotting.plot_maps_qc(file_hash, selected_timestamp))
