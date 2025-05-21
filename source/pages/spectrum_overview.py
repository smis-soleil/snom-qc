"""
This pages shows a spectrum QC report for the uploaded document.
"""

from io import BytesIO
import streamlit as st
from anasyspythontools import export
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import streamlit_shortcuts
import pandas as pd
import numpy as np
import xarray as xr

import utils
from utils import SessionStateSingleton

# Set up sidebar (navigation and uploader to change file)
utils.setup_page_with_redirect(allowed_file_types=['axz', 'axd'])

st.title('Spectrum overview')

st.write(
    '''
    This page shows an overview of the spectra in the uploaded document. By default, all spectra and channels are shown, but this can be customised below. 
    '''
)

def button_download_qc(placeholder, key):
    if 'all' in st.session_state.cached_spectrum_qa:
        buttondata = st.session_state.cached_spectrum_qa['all']
        buttondisabled = False
        buttontext = 'Download QC report'
    else:
        buttondata = b''
        buttondisabled = True
        buttontext = 'Generating QC report...'
    
    placeholder.download_button(
        label=buttontext,
        data=buttondata,
        file_name=f'{SessionStateSingleton().get_file_name().replace(".", "_")}_spectrumqc.pdf',
        type='primary',
        icon=':material/picture_as_pdf:',
        disabled=buttondisabled,
        key=key,
    )

def button_download_csv(placeholder, key):
    if st.session_state.cached_spectrum_csv is not None:
        buttondata = st.session_state.cached_spectrum_csv
        buttondisabled = False
        buttontext = 'Download all spectra as CSV'
    else:
        buttondata = b''
        buttondisabled = True
        buttontext = 'Generating CSV export...'
    
    placeholder.download_button(
        label=buttontext,
        data=buttondata,
        file_name=f'{SessionStateSingleton().get_file_name().replace(".", "_")}_spectrumdata.csv',
        type='secondary',
        icon=':material/table_view:',
        disabled=buttondisabled,
        key=key
    )
col1, col2 = st.columns(2, gap='small')

download_placeholder = col1.empty()
csv_placeholder = col2.empty()
button_download_qc(download_placeholder, key = 2)
button_download_csv(csv_placeholder, 3)

# Get all spectrum labels and data channels
filename = SessionStateSingleton().get_file_name()
doc = SessionStateSingleton().get_anasys_doc()
spectrum_labels, data_channels_available = utils.get_list_of_spectra_and_data_channels(doc)
data_channels_available = utils.sort_spectrum_datachannels(data_channels_available)

# Main content
if len(spectrum_labels) == 0:
    st.error('No spectra found. Choose another file')
    st.stop()

# Other options
with st.expander('Display settings and metadata', expanded=False):

    st.write('**Display settings**')
    st.caption('Channels to display')
    show_channels = st.multiselect('Choose channels to display:', data_channels_available, default=data_channels_available, label_visibility='collapsed')

    st.caption('Select map to display')
    DEFAULT_MAP_OPTION = 'Default (most recent map before spectrum acquisition)'
    map_to_show = st.selectbox(
        label='Choose map to display:',
        options=[DEFAULT_MAP_OPTION, *doc.HeightMaps],
        index=0,
        label_visibility='collapsed'
    )
    map_to_show = None if map_to_show == DEFAULT_MAP_OPTION else map_to_show

    st.caption('Other options')
    show_other_spectra = st.checkbox('Show all spectra in file', value=True)
    show_spectrum_labels = st.checkbox('Show spectrum labels on image', value=True)
    

    download_button_container = st.empty()

    # Show metadata
    metadata_df = SessionStateSingleton().get_cached_spectrum_metadata()
    st.write('**Spectrum metadata**')
    st.caption('Select properties to display')
    selected_metadata_tags = st.multiselect(
        label='Metadata properties to display',
        options=metadata_df.columns,
        default=[
            'Label', 'TimeStamp', 'Location.X', 'Location.Y', 'StartWavenumber', 
            'EndWavenumber', 'PulseRate', 'Power', 'OAPFocusPos', 'Background.ID', 
            # 'BackgroundPower', 'Background.IRSweepResolution',
        ],
        label_visibility='collapsed'
    )
    # st.caption('Metadata table. Select row to display the corresponding map')
    metadata_table = st.dataframe(
        metadata_df, selection_mode='multi-row', on_select='ignore', hide_index=True,
        column_order=selected_metadata_tags,
        use_container_width=True,
    )


# st.divider()
if len(spectrum_labels) == 1:
    highlight_spectrum = spectrum_labels[0]

else:
    # Spectrum selection
    highlight_spectrum = st.select_slider(
        label='Choose spectrum to highlight',
        options=spectrum_labels,
        label_visibility='collapsed',
        value=spectrum_labels[0]
    )

st.empty().pyplot(SessionStateSingleton().get_cached_spectrum_qa(
    highlight_spectrum, show_channels, show_other_spectra, 
    map_to_show, show_spectrum_labels
))

# If the report files were not generated yet, do that and replace the buttons
SessionStateSingleton().get_cached_spectrum_qa_pdf()
button_download_qc(download_placeholder, key = 1)
SessionStateSingleton().get_cached_spectrum_csv()
button_download_csv(csv_placeholder, key = 4)