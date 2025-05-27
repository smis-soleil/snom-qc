import streamlit as st
from utils import SessionStateSingleton
import utils

from anasyspythontools import export
import xarray as xr


utils.setup_page_with_redirect(['axz', 'axd'])
st.page_link(
    'pages/spectrum_overview.py',
    label='Back to spectrum overview',
    icon=':material/arrow_back:',
)

doc = SessionStateSingleton().get_anasys_doc()
spectrum_labels, data_channels_available = utils.get_list_of_spectra_and_data_channels(doc)
data_channels_available = utils.sort_spectrum_datachannels(data_channels_available)

spectra = xr.concat(dim = 'si', objs = [ 
    export.spectrum_to_Dataset(s)
    for s in doc.RenderedSpectra.values()
])

channel = st.selectbox(
    label='Choose channel to export',
    options=data_channels_available,
    index=0,
    label_visibility='visible'
)

spectra = spectra[channel]

spectra_df = spectra.to_pandas().assign(
    Label = spectra.Label.values,
    X = spectra['Location.X'].values,
    Y = spectra['Location.Y'].values,
).set_index(['Label', 'X', 'Y'])


dlname = st.text_input(
    label='File name for download',
    value = f'{SessionStateSingleton().get_file_name().replace('.','_')}_{channel.replace(' ','_')}.csv'
)

st.download_button(
    label='Download CSV file',
    data=spectra_df.to_csv(),
    file_name=dlname,
    mime='text/csv',
    help='Download the spectra as a CSV file for further processing.',
    icon=':material/table_view:',
    key='download_spectra_csv',
    type='primary',
)

st.dataframe(spectra_df, use_container_width=True)