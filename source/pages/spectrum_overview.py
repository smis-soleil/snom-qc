"""
This pages shows a spectrum QC report for the uploaded document.
"""

from io import BytesIO
import streamlit as st
from anasyspythontools import export
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import streamlit_shortcuts

import utils
from plotting import plot_spectrum_qc   # pylint: disable=E0401

# Set up sidebar (navigation and uploader to change file)
utils.setup_page_with_redirect(allowed_file_types=['axz', 'axd'])

st.write('**TODO:** check bug with 2024-07-24_TB012F_d2t56_maps.axz')
st.write('**TODO:** display spectrum metadata')

# Get all spectrum labels and data channels
filename = st.session_state.file_name
doc = st.session_state.anasys_doc
spectrum_labels, data_channels_available = utils.get_list_of_spectra_and_data_channels(doc)

# Main content
if len(spectrum_labels) == 0:
    st.error('No spectra found. Choose another file')
    st.stop()

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

# Other options
with st.expander('Display settings', expanded=True):

    cols = st.columns(3, vertical_alignment='bottom', gap='large')
    show_other_spectra = cols[1].checkbox('Show other spectra', value=True)
    show_spectrum_labels = cols[2].checkbox('Show spectrum labels', value=True)

    DEFAULT_MAP_OPTION = 'Default (most recent map before spectrum acquisition)'
    map_to_show = cols[0].selectbox(
        label='Choose map to display:',
        options=[DEFAULT_MAP_OPTION, *doc.HeightMaps],
        index=0
    )
    map_to_show = None if map_to_show == DEFAULT_MAP_OPTION else map_to_show
    
    show_channels = st.multiselect('Choose channels to display:', data_channels_available, default=data_channels_available)

st.pyplot(plot_spectrum_qc(doc, highlight_spectrum, show_channels, show_other_spectra, map_to_show, show_spectrum_labels))

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


st.download_button(
    label='Download QC report as PDF',
    data=generate_pdf(),
    file_name=f'{filename}_spectrumqc.pdf',
    type='primary'
)