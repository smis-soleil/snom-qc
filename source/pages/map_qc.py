"""
This pages shows a map QC report for the uploaded document.
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from utils import setup_page_with_redirect, collect_map_qc_warnings
import plotting  # pylint: disable=E0401

# Set up sidebar (navigation and uploader to change file)
setup_page_with_redirect(allowed_file_types=['axz', 'axd'])

st.title('Map QC report')
st.write(
    '''
    This page shows a map QC report for the uploaded document. The report includes a visualisation of all maps at a given timestamp, as well as a download link for a PDF version of the report. We also check for common issues such as high fluctutions in the deflection or IR phase signal, PLL saturation, and data processing errors. This is based on simple heuristics and should not be considered a definitive QC report.
    '''
)

# Load map properties from uploaded file
@st.cache_resource
def list_heightmap_groups(file_hash):
    """
    Function to load map properties from uploaded file
    """

    doc = st.session_state.anasys_doc
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

file_hash = st.session_state.file_hash
doc_uploaded, map_df = list_heightmap_groups(file_hash)

# Main content
if len(doc_uploaded.HeightMaps) == 0:
    st.error('No maps found. Choose another file')
    st.stop()

# Choose maps to display
st.caption('Choose maps to display:')
hmap_groups_table = st.dataframe(
    map_df,
    selection_mode='single-row', on_select='rerun', hide_index=True,
    use_container_width=True
)

# Get selected timestamp
selected_indices = hmap_groups_table['selection']['rows']
selected_index = 0 if len(selected_indices) == 0 else selected_indices[0]
selected_timestamp = map_df.index[selected_index]

# Plot QC maps
with st.spinner('Generating QC report...'):

    st.download_button(
        label='Download QC report as PDF',
        data=plotting.generate_mapqc_pdf(file_hash),
        file_name=f'{st.session_state.file_name}_mapqc.pdf',
        type='primary'
    )

with st.spinner('Plotting data...'):
    st.pyplot(plotting.plot_maps_qc(file_hash, selected_timestamp))

map_warnings = collect_map_qc_warnings(doc_uploaded, map_df.loc[selected_timestamp, 'Map Labels'])
if len(map_warnings) > 0:
    st.warning(icon=':material/warning:', body='The following potential issues were detected:\n\n* ' + '\n\n* '.join(map_warnings))