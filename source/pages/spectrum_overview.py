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

import utils
from plotting import plot_spectrum_qc, sort_spectrum_datachannels   # pylint: disable=E0401

# Set up sidebar (navigation and uploader to change file)
utils.setup_page_with_redirect(allowed_file_types=['axz', 'axd'])

st.title('Spectrum overview')

st.write(
    '''
    This page shows an overview of the spectra in the uploaded document. By default, all spectra and channels are shown, but this can be customised below. 
    '''
)

download_placeholder = st.empty()
download_placeholder.download_button(
    label='Generating PDF report',
    data=b'1',
    # file_name=f'{filename}_spectrumqc.pdf',
    icon=':material/download:',
    type='primary',
    disabled=True,
    key='download_button_disabled'
)

# TODO: check bug with 2024-07-24_TB012F_d2t56_maps.axz
# TODO: display spectrum metadata

# Get all spectrum labels and data channels
filename = st.session_state.file_name
doc = st.session_state.anasys_doc
spectrum_labels, data_channels_available = utils.get_list_of_spectra_and_data_channels(doc)
data_channels_available = sort_spectrum_datachannels(data_channels_available)

# Main content
if len(spectrum_labels) == 0:
    st.error('No spectra found. Choose another file')
    st.stop()


# Other options
with st.expander('Display settings and metadata', expanded=False):

    st.write('**Display settings**')
    st.caption('Channels to display')
    show_channels = st.multiselect('Choose channels to display:', data_channels_available, default=data_channels_available, label_visibility='collapsed')
    # cols = st.columns(3, vertical_alignment='bottom', gap='large')

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
    metadata_df = utils.parse_spectrum_metadata(st.session_state.file_hash)
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


@st.cache_resource
def generate_pdf():
    """
    Generate a PDF file of the QC report, using an in_memory BytesIO buffer
    """

    with BytesIO() as f:
        with PdfPages(f) as pdf:
            for l in spectrum_labels:
                fig = plot_spectrum_qc(doc, l, channels_to_show=data_channels_available, show_other_spectra=False, show_spectrum_labels=False)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

        return f.getvalue()


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

st.empty().pyplot(plot_spectrum_qc(doc, highlight_spectrum, show_channels, show_other_spectra, map_to_show, show_spectrum_labels))


download_placeholder.download_button(
    label='Download QC report as PDF',
    data=generate_pdf(),
    file_name=f'{filename}_spectrumqc.pdf',
    icon=':material/download:',
    type='primary'
)

st.warning('**TODO** QC control')