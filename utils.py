"""
Utility file for streamlit app

Mainly takes care of setting up the sidebar and handling file uploads
Use it by calling `setup_page()` in the main app file and 
`setup_page_with_redirect()` in other pages
"""

import gzip
import hashlib
import xml.etree.ElementTree as ET  # for parsing XML
import pandas as pd
from anasyspythontools import anasysio, irspectra, anasysdoc

import streamlit as st


def setup_page(streamlit_layout='wide'):
    """ 
    Set up a page's sidebar with navitation links and file upload widget
    """

    st.set_page_config(layout=streamlit_layout)

    # Make sure all keys are initialised (to None if need be)
    # and get their values
    initialise_session_state()
    full_file = st.session_state.file_extension in ['axz', 'axd']

    with st.sidebar:

        # Set up navigation links
        st.page_link("app.py", label="Home")
        st.page_link(
            'pages/map_overview.py',
            label='Map overview',
            disabled=not full_file)
        st.page_link(
            'pages/map_qc.py',
            label='Map QC',
            disabled=not full_file)
        st.page_link(
            'pages/spectrum_overview.py',
            label='Display spectra',
            disabled=not full_file)
        st.page_link(
            'pages/test_spectrum_plotly.py',
            label='Test spectrum plotly',
        )

        st.divider()

        initialise_upload_widget()


def setup_page_with_redirect(allowed_file_types, streamlit_layout='wide'):
    """
    Checks whether the file uploaded is of the correct type to view a page
    Redirects to home page if not
    """

    initialise_session_state()
    if st.session_state.file_extension not in allowed_file_types:
        # if st.session_state.file_extension == 'irb':
        #     st.switch_page('pages/background_overview.py')
        st.switch_page('app.py')

    # Defer further setup to setup_page()
    setup_page(streamlit_layout=streamlit_layout)

def initialise_session_state():
    """
    Initialise session state variables
    """

    if 'anasys_doc' not in st.session_state:
        st.session_state.anasys_doc = None

    if 'file_name' not in st.session_state:
        st.session_state.file_name = None

    if 'file_extension' not in st.session_state:
        st.session_state.file_extension = None

    if 'file_upload_widget_key' not in st.session_state:
        st.session_state.file_upload_widget_key = 'upload_widget'

def parse_file(file):
    """
    Parse anasys file and store it in session state
    """

    f_hash = get_file_hash(file)
    f_data = gzip.open(file) if file.name.endswith('.axz') else file
    f_data = ET.iterparse(f_data)
    f_data = anasysio.AnasysFileReader._strip_namespace(None, f_data)  # pylint: disable=W0212
    f_data = f_data.root

    if file.name.endswith('irb'):
        doc = irspectra.Background(f_data)
    else:
        doc = anasysdoc.AnasysDoc(f_data)

    st.session_state.anasys_doc = doc
    st.session_state.file_name = file.name
    st.session_state.file_hash = f_hash
    st.session_state.file_extension = file.name.split('.')[-1]


def upload_example_file():
    """
    Uploads example file
    """

    # Changing the key of the upload widget resets it to an empty state on a script rerun
    # Effectively, this clears the previously uploaded file
    st.session_state.file_upload_widget_key += "'"

    with open('example.axz', 'rb') as f:
        parse_file(f)

def initialise_upload_widget():
    """
    Set up file uploader widget, including
    - currently uploaded filename (or call to action)
    - upload widget
    - button to load example file
    """

    if st.session_state.file_name is not None:
        st.info(
            st.session_state.file_name,
            # + '\n' + st.session_state.file_hash,
            icon=':material/description:'
        )
    else:
        st.error('Upload file here')

    def on_upload():
        file = st.session_state[st.session_state.file_upload_widget_key]

        # When the uploader is cleared by the user, `file` will be None
        if file is not None:
            parse_file(file)

    st.file_uploader('Upload a file', type=['axz', 'axd'], label_visibility='collapsed',
                     on_change=on_upload, key=st.session_state.file_upload_widget_key)

    st.button('Load example file', on_click=upload_example_file)

def get_file_hash(file):
    """
    Get the md5 hash of a file.
    """

    hash_md5 = hashlib.md5()
    for chunk in iter(lambda: file.read(4096), b""):
        hash_md5.update(chunk)

    # Reset file pointer to beginning
    file.seek(0)
    return hash_md5.hexdigest()


@st.cache_resource
def parse_map_metadata(file_hash):  # pylint: disable=W0613
    """
    Function to load map metadata from uploaded file.
    The file hash is used to cache the result.
    """

    # Grab document from session state
    doc = st.session_state.anasys_doc

    # Define map of tags to display by default, and functions to extract them
    map_important_tags = {
        'Label':         lambda k, _: k,
        'Timestamp':     lambda _, v: pd.to_datetime(v.TimeStamp).strftime('%Y-%m-%d %H:%M:%S'),
        'DataChannel':   lambda _, v: v['DataChannel'],
        'Units':         lambda _, v: v['UnitPrefix'] + v['Units'],
        'IRWavenumber':  lambda _, v: v.Tags['IRWavenumber'],
        'ScanRate':      lambda _, v: v.Tags['ScanRate'],
        'IRAttenuation': lambda _, v: v.Tags['IRAttenuation'],
        'dx':            lambda _, v: v.SampleBase64.shape[1],
        'dX':            lambda _, v: f'{v.Size.X} µm',
        "IRPulseRate":   lambda _, v: v.Tags['IRPulseRate'],
        "IRPulseWidth":  lambda _, v: v.Tags['IRPulseWidth'],
        "TraceRetrace":  lambda _, v: v.Tags['TraceRetrace'],
        'dy':            lambda _, v: v.SampleBase64.shape[0],
        'dY':            lambda _, v: f'{v.Size.Y} µm',
    }

    # List all other tags in the objects
    map_other_tags = {
        k: (lambda tag: lambda x, v: v.Tags[tag])(k)  # pylint: disable=C3002
        for k in list(doc.HeightMaps.values())[0].Tags
        if k not in map_important_tags
    }

    # Combine all maps
    combined_map = {**map_important_tags, **map_other_tags}

    # Create a dataframe of all maps and their properties
    df = pd.DataFrame([
        {
            tagname: tagfunc(maplabel, hmap)
            for tagname, tagfunc in combined_map.items()
        }
        for maplabel, hmap in doc.HeightMaps.items()
    ])

    return doc, df


from anasyspythontools import export

def get_list_of_spectra_and_data_channels(doc):
    """
    Get all spectrum labels and data channels
    """

    spectrum_labels = list(doc.RenderedSpectra)
    all_channels = []
    for spectrum in doc.RenderedSpectra.values():
        for k in export.spectrum_to_Dataset(spectrum).keys():
            if k not in all_channels:
                all_channels.append(k)

    return spectrum_labels, all_channels    