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
from anasyspythontools import anasysio, irspectra, anasysdoc, anasysfile
import numpy as np

import streamlit as st

import os

DEV_MODE = os.environ.get('DEV_MODE', False)

class SessionState:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SessionState, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        if 'anasys_doc' not in st.session_state:
            st.session_state.anasys_doc = None

        if 'file_name' not in st.session_state:
            st.session_state.file_name = None

        if 'file_extension' not in st.session_state:
            st.session_state.file_extension = None

        if 'file_hash' not in st.session_state:
            st.session_state.file_hash = None

        if 'file_upload_widget_key' not in st.session_state:
            st.session_state.file_upload_widget_key = 'upload_widget'

        if 'error_message' not in st.session_state:
            st.session_state.error_message = None

    def get_anasys_doc(self):
        self._initialize()
        return st.session_state.anasys_doc

    def get_file_name(self):
        self._initialize()
        return st.session_state.file_name

    def get_file_hash(self):
        self._initialize()
        return st.session_state.file_hash
    
    def get_file_extension(self):
        self._initialize()
        return st.session_state.file_extension

    def get_file_upload_widget_key(self):
        self._initialize()
        return st.session_state.file_upload_widget_key
    
    def get_error_message(self):
        self._initialize()
        return st.session_state.error_message
    
    def get_upload_widget(self):
        self._initialize()
        return st.session_state[st.session_state.file_upload_widget_key]

    def set_anasys_doc(self, doc, fname, fhash, fext):
        self._initialize()
        st.session_state.anasys_doc = doc
        st.session_state.file_name = fname
        st.session_state.file_hash = fhash
        st.session_state.file_extension = fext

        # Clear error state, if any
        st.session_state.error_message = None

    def set_file_upload_widget_key(self, value):
        self._initialize()
        st.session_state.file_upload_widget_key = value

    def set_error_message(self, value):
        self._initialize()
        st.session_state.anasys_doc = None
        st.session_state.file_name = None
        st.session_state.file_hash = None
        st.session_state.file_extension = None
        st.session_state.error_message = value


def setup_page(streamlit_layout='centered'):
    """ 
    Set up a page's sidebar with navitation links and file upload widget
    """

    st.set_page_config(
        page_title='Anasys Python Tools' + (' (dev)' if DEV_MODE else ''),
        layout=streamlit_layout
    )

    # Make sure all keys are initialised (to None if need be)
    # and get their values
    ss = SessionState()


    full_file = ss.get_file_extension() in ['axz', 'axd']

    with st.sidebar:

        if DEV_MODE:
            st.error('Development mode', icon=':material/warning:')

        # Set up navigation links
        st.page_link("app.py", label="Home")
        st.page_link(
            'pages/optical.py',
            label='Optical images',
            disabled=not full_file)
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

        st.divider()

        initialise_upload_widget()


def setup_page_with_redirect(allowed_file_types, streamlit_layout='centered'):
    """
    Checks whether the file uploaded is of the correct type to view a page
    Redirects to home page if not
    """

    
    if SessionState().get_file_extension() not in allowed_file_types:
        # if st.session_state.file_extension == 'irb':
        #     st.switch_page('pages/background_overview.py')
        st.switch_page('app.py')

    # If page in error state, redirect to home page
    if SessionState().get_error_message():
        st.switch_page('app.py')


    # Defer further setup to setup_page()
    setup_page(streamlit_layout=streamlit_layout)


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

    SessionState().set_anasys_doc(doc, file.name, f_hash, file.name.split('.')[-1])


def upload_example_file():
    """
    Uploads example file
    """

    # Changing the key of the upload widget resets it to an empty state on a script rerun
    # Effectively, this clears the previously uploaded file
    SessionState().set_file_upload_widget_key(
        SessionState().get_file_upload_widget_key() + "1"
    )

    with open('source/example.axz', 'rb') as f:
        parse_file(f)

def initialise_upload_widget():
    """
    Set up file uploader widget, including
    - currently uploaded filename (or call to action)
    - upload widget
    - button to load example file
    """

    if SessionState().get_file_name() is not None:
        st.info(
            SessionState().get_file_name(),
            # + '\n' + st.session_state.file_hash,
            icon=':material/description:'
        )
    else:
        st.error('Upload file here')

    def on_upload():

        try:
            file = SessionState().get_upload_widget()
            # When the uploader is cleared by the user, `file` will be None
            if file is not None:
                parse_file(file)

        except Exception as e:
            print(f'Error encountered on file upload: {e}')
            SessionState().set_error_message('Error encountered on file upload. Is this a valid Anasys file?')
    
    st.file_uploader('Upload a file', type=['axz', 'axd'], label_visibility='collapsed',
                     on_change=on_upload, key=SessionState().get_file_upload_widget_key())

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
    doc = SessionState().get_anasys_doc()

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

@st.cache_resource
def parse_spectrum_metadata(file_hash):  # pylint: disable=W0613
    doc = SessionState().get_anasys_doc()

    # Define map of tags to display by default, and functions to extract them
    spectrum_important_tags = {
        'Label':         lambda k, _: k,
    }

    df = (
        pd.DataFrame([doc.RenderedSpectra[k].attrs for k in doc.RenderedSpectra])
        .assign(
            Power = lambda x: x.AttenuationBase64.map(np.mean)*100,
            BackgroundPower = lambda x: x['Background.AttenuatorPower'].map(np.mean)*100,
            TimeStamp = lambda x: x.TimeStamp.map(lambda x: pd.to_datetime(x).strftime('%Y-%m-%d %H:%M:%S'))
        )
        .drop(columns=[
            'AttenuationBase64', 'Background.AttenuatorPower', 'Background.Table',
            "Background.UnitOffset", "Background.UnitScale", "Background.Units", 
            "Background.signal", 'Background.wn', 'Background', 'BeamShapeFactorBase64',
            'ChannelZero', "DataChannels.IR Phase (Deg)", "DataChannels.PLL Frequency (kHz)", 
            "DataChannels.IR Amplitude (mV)", 'DutyCycle', 'FreqWindowData',
            'RotaryPolarizerMotorPositionBase64', 'Background.PolarizerPosition',
        ], errors='ignore')
    )

    df = df.drop(columns=[c for c in df.columns if 'XMLSchema-instance' in c])
    # map all values of type AnasysElement to None
    df = df.applymap(lambda x: None if isinstance(x, anasysfile.AnasysElement) else x)
    
    # reorder columns to get label and timestamp first
    df = df[['Label', 'TimeStamp'] + [c for c in df.columns if c not in ['Label', 'TimeStamp']]]
    return df


from anasyspythontools import export

def get_list_of_spectra_and_data_channels(doc):
    """
    Get all spectrum labels and data channels
    """

    spectrum_labels = list(doc.RenderedSpectra)
    # print(doc.HeightMaps)

    all_channels = []
    for spectrum in doc.RenderedSpectra.values():
        for k in export.spectrum_to_Dataset(spectrum).keys():
            if k not in all_channels:
                all_channels.append(k)

    return spectrum_labels, all_channels    

def collect_map_qc_warnings(doc, mapkeys):
    flatten = lambda l: [item for sublist in l for item in sublist]
    return flatten([map_qc_check(doc.HeightMaps[key]) for key in mapkeys])

def map_qc_check(hmap):
    warnings = []

    if hmap.DataChannel == 'deflection':
        # Check that deflection raw values are reported (mean is close to setpoint)
        defmean = np.mean(hmap.SampleBase64)
        setpoint = float(hmap.Tags['Setpoint'].split(' ')[0])
        if DEV_MODE or np.abs(defmean - setpoint) > 0.1:
            warnings.append(
                f'Possible data processing of deflection map ({hmap.Label}): the mean value of {defmean:.3f} V is far from the recorded setpoint of {setpoint:.3f} V'
            )

        # Check deflction fluctuations are not too high
        setpoint = float(hmap.Tags['Setpoint'].split(' ')[0])
        frac_above_0_01 = np.sum(np.abs(hmap.SampleBase64 - setpoint) > 0.01) / hmap.SampleBase64.size
        if DEV_MODE or frac_above_0_01 > .01:
            warnings.append(
                f'Bad AFM tracking in deflection map ({hmap.Label}): {frac_above_0_01*100:.1f}% of pixels is over 0.01 V away from the setpoint of {setpoint:.3f} V'
            )
    if hmap.DataChannel == 'phase2':
        # Check phase fluctuations are not too high
        frac_above_20 = np.sum(np.abs(hmap.SampleBase64) > 20)/ hmap.SampleBase64.size
        if DEV_MODE or frac_above_20 > .01:
            warnings.append(
                f'Bad IR tracking in IR Phase map ({hmap.Label}): {frac_above_20*100:.1f}% of pixels is above 20°'
            )
        # Check high mean
        phasemean = np.mean(hmap.SampleBase64)
        if DEV_MODE or np.abs(phasemean) > 1:
            warnings.append(
                f'Bad IR tracking in IR Phase map ({hmap.Label}): the mean value of {phasemean:.2f}° is far from 0°'
            )
    if hmap.DataChannel == 'freq2':
        # Check for PLL saturation
        data = hmap.SampleBase64
        pllmax = np.sum(np.abs(np.max(data) - data) < .0001) / data.size
        pllmin = np.sum(np.abs(np.min(data) - data) < .0001) / data.size
        if DEV_MODE or pllmax > .1 or pllmin > 0.1:
            warnings.append(
                f'Saturation in PLL Frequency map ({hmap.Label}): {pllmax:.1f}% ({pllmin:.1f}%) of pixels are equal to the image maximum (minimum)'
            )
        # Check that PLL is not processed
        pllmean = np.mean(data)
        if DEV_MODE or np.abs(pllmean) < 10:
            warnings.append(
                f'Possible data processing of PLL Frequency map ({hmap.Label}): the mean value of {pllmean:.2f} kHz is close to zero'
            )

    if hmap.DataChannel == 'height':
        lines_close_to_zero = np.sum(np.mean(np.abs(hmap.SampleBase64), axis =1) < 10) / hmap.SampleBase64.shape[0]
        if DEV_MODE or lines_close_to_zero > .01:
            warnings.append(
                f'Possible data processing of height map ({hmap.Label}): {lines_close_to_zero:.1f}% of lines have a mean value below 10 nm'
            )

    # Check that IR amplitude is reported without processing
    return warnings