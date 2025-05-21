import streamlit as st
import matplotlib.pyplot as plt
from utils import SessionStateSingleton

from anasyspythontools import export

st.set_page_config(layout="centered")

from utils import setup_page_with_redirect, initialise_upload_widget
setup_page_with_redirect(allowed_file_types=['axz', 'axd', 'irb'])

st.write(SessionStateSingleton().get_file_name())

doc = SessionStateSingleton().get_anasys_doc()

if SessionStateSingleton().get_file_extension() == 'irb':
    wavenumbers = doc.wn
    signal = doc.Table
    fig, ax = plt.subplots()
    ax.plot(wavenumbers, signal)
    ax.invert_xaxis()
    ax.set_xlabel('Wavenumber (cm⁻¹)')
    ax.set_ylabel('Laser power (mW)')
    st.pyplot(fig)

else:
    spectra = export.spectra_list_to_Dataset(doc.RenderedSpectra.values())
    st.write('TODO')