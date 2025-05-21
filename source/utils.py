"""
Utility file for streamlit app

Mainly takes care of setting up the sidebar and handling file uploads
Use it by calling `setup_page()` in the main app file and 
`setup_page_with_redirect()` in other pages
"""

import colorsys
from io import BytesIO
import os
import gzip
import hashlib
import xml.etree.ElementTree as ET  # for parsing XML
import pandas as pd
from anasyspythontools import anasysio, irspectra, anasysdoc, anasysfile, export
import numpy as np
import xarray as xr

# PLotting imports
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from matplotlib.colors import to_rgb
from matplotlib.backends.backend_pdf import PdfPages

import streamlit as st

DEV_MODE = os.environ.get('DEV_MODE', False)

class SessionStateSingleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SessionStateSingleton, cls).__new__(cls)
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

        if 'cached_image_qa' not in st.session_state:
            st.session_state.cached_image_qa = {}

        if 'cached_spectrum_qa' not in st.session_state:
            st.session_state.cached_spectrum_qa = {}

        if 'cached_spectrum_csv' not in st.session_state:
            st.session_state.cached_spectrum_csv = None

        if 'cached_map_metadata' not in st.session_state:
            st.session_state.cached_map_metadata = None

        if 'cached_spectrum_metadata' not in st.session_state:
            st.session_state.cached_spectrum_metadata = None

        if 'cached_heightmap_metadata' not in st.session_state:
            st.session_state.cached_heightmap_metadata = None

        if 'cached_heightmap_groups' not in st.session_state:
            st.session_state.cached_heightmap_groups = None

    def get_cached_heightmap_groups(self):
        self._initialize()

        if st.session_state.cached_heightmap_groups is None:
            # Get the map metadata
            st.session_state.cached_heightmap_groups = \
                list_heightmap_groups(self.get_anasys_doc())

        return st.session_state.cached_heightmap_groups

    def get_cached_heightmap_metadata(self):
        self._initialize()

        if st.session_state.cached_heightmap_metadata is None:
            # Get the map metadata
            st.session_state.cached_heightmap_metadata = \
                parse_map_metadata(self.get_anasys_doc())
        
        return st.session_state.cached_heightmap_metadata
    
    def get_cached_spectrum_metadata(self):
        self._initialize()

        if st.session_state.cached_spectrum_metadata is None:
            # Get the spectrum metadata
            st.session_state.cached_spectrum_metadata = \
                parse_spectrum_metadata(self.get_anasys_doc())

        return st.session_state.cached_spectrum_metadata
    
    def get_cached_image_qa(self, timestamp=None, ncols=None, allmaps=False):
        self._initialize()
        
        if not allmaps:
            key = (timestamp, ncols)
            if key not in st.session_state.cached_image_qa:
                # Get the map metadata
                st.session_state.cached_image_qa[key] = \
                    plot_maps_qc_at_timestamp(timestamp, ncols=ncols)
            return st.session_state.cached_image_qa[key]
        
        if 'allmaps' not in st.session_state.cached_image_qa:
            st.session_state.cached_image_qa['allmaps'] = \
                generate_mapqc_pdf(self.get_file_hash(), ncols=ncols)
            
        return st.session_state.cached_image_qa['allmaps']
    
    def get_cached_spectrum_qa(self, selected_label, channels, show_others, show_map, show_labels):
        self._initialize()

        key = (selected_label, ''.join(channels), show_others, show_map, show_labels)
        if key not in st.session_state.cached_spectrum_qa:
            # Get the spectrum metadata
            st.session_state.cached_spectrum_qa[key] = \
                plot_spectrum_qc(self.get_anasys_doc(), selected_label, channels, show_others, show_map, show_labels)

        return st.session_state.cached_spectrum_qa[key]
    
    def get_cached_spectrum_qa_pdf(self):
        self._initialize()
        
        if 'all' not in st.session_state.cached_spectrum_qa:
            doc = self.get_anasys_doc()
            spectrum_labels, data_channels_available = get_list_of_spectra_and_data_channels(doc)

            with BytesIO() as f:
                with PdfPages(f) as pdf:
                    for l in spectrum_labels:
                        fig = self.get_cached_spectrum_qa(l, channels=data_channels_available, show_others=False, show_map=None, show_labels=False)
                        pdf.savefig(fig, bbox_inches='tight')
                        plt.close(fig)

                st.session_state.cached_spectrum_qa['all'] = f.getvalue()
                
        return st.session_state.cached_spectrum_qa['all']
    
    def get_cached_spectrum_csv(self):
        self._initialize()
        
        if st.session_state.cached_spectrum_csv is None:
                doc = self.get_anasys_doc()
                spectrum_labels, data_channels_available = get_list_of_spectra_and_data_channels(doc)
                data_channels_available = sort_spectrum_datachannels(data_channels_available)

                spectra = xr.concat(dim = 'si', objs = [ 
                    export.spectrum_to_Dataset(s)
                    for s in self.get_anasys_doc().RenderedSpectra.values()
                ])

                spectra = spectra[data_channels_available[0]]

                spectra_df = spectra.to_pandas().assign(
                    Label = spectra.Label.values,
                    X = spectra['Location.X'].values,
                    Y = spectra['Location.Y'].values,
                ).set_index(['Label', 'X', 'Y'])

                with BytesIO() as buffer:
                    spectra_df.to_csv(buffer)
                    st.session_state.cached_spectrum_csv = buffer.getvalue()

        return st.session_state.cached_spectrum_csv

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

        if fhash != st.session_state.file_hash:
            st.session_state.anasys_doc = doc
            st.session_state.file_name = fname
            st.session_state.file_hash = fhash
            st.session_state.file_extension = fext

            st.session_state.cached_image_qa = {}
            st.session_state.cached_spectrum_qa = {}
            st.session_state.cached_spectrum_csv = None
            st.session_state.cached_map_metadata = None
            st.session_state.cached_spectrum_metadata = None
            st.session_state.cached_heightmap_metadata = None
            st.session_state.cached_heightmap_groups = None

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

    st.markdown(unsafe_allow_html=True, body="""
        <style>
            div[data-testid="stColumn"] {
                width: fit-content !important;
                flex: unset;
            }
            div[data-testid="stColumn"] * {
                width: fit-content !important;
            }
        </style>
    """)

    # Make sure all keys are initialised (to None if need be)
    # and get their values
    ss = SessionStateSingleton()

    full_file = ss.get_file_extension() in ['axz', 'axd']

    with st.sidebar:

        if DEV_MODE:
            st.error('Development mode', icon=':material/warning:')

        # Set up navigation links
        st.page_link("app.py", label="Home")
        st.page_link(
            'pages/optical.py',
            label='Optical Images',
            disabled=not full_file)
        st.page_link(
            'pages/map_overview.py',
            label='Map Overview',
            disabled=not full_file)
        st.page_link(
            'pages/map_qc.py',
            label='Map QC',
            disabled=not full_file)
        st.page_link(
            'pages/spectrum_overview.py',
            label='Spectra',
            disabled=not full_file)

        st.divider()

        initialise_upload_widget()


def setup_page_with_redirect(allowed_file_types, streamlit_layout='centered'):
    """
    Checks whether the file uploaded is of the correct type to view a page
    Redirects to home page if not
    """

    
    if SessionStateSingleton().get_file_extension() not in allowed_file_types:
        # if st.session_state.file_extension == 'irb':
        #     st.switch_page('pages/background_overview.py')
        st.switch_page('app.py')

    # If page in error state, redirect to home page
    if SessionStateSingleton().get_error_message():
        st.switch_page('app.py')


    # Defer further setup to setup_page()
    setup_page(streamlit_layout=streamlit_layout)


def parse_file(file):
    """
    Parse anasys file and store it in session state
    """

    f_hash = get_file_hash(file)

    if SessionStateSingleton().get_file_hash() == f_hash:
        return

    f_data = gzip.open(file) if file.name.endswith('.axz') else file
    f_data = ET.iterparse(f_data)
    f_data = anasysio.AnasysFileReader._strip_namespace(None, f_data)  # pylint: disable=W0212
    f_data = f_data.root

    if file.name.endswith('irb'):
        doc = irspectra.Background(f_data)
    else:
        doc = anasysdoc.AnasysDoc(f_data)

    display_name = file.name.split('/')[-1]

    SessionStateSingleton().set_anasys_doc(doc, display_name, f_hash, file.name.split('.')[-1])


def upload_example_file():
    """
    Uploads example file
    """

    # Changing the key of the upload widget resets it to an empty state on a script rerun
    # Effectively, this clears the previously uploaded file
    SessionStateSingleton().set_file_upload_widget_key(
        SessionStateSingleton().get_file_upload_widget_key() + "1"
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

    if SessionStateSingleton().get_file_name() is not None:
        st.info(
            SessionStateSingleton().get_file_name(),
            # + '\n' + st.session_state.file_hash,
            icon=':material/description:'
        )
    else:
        st.error('Upload file here')

    def on_upload():

        try:
            file = SessionStateSingleton().get_upload_widget()
            # When the uploader is cleared by the user, `file` will be None
            if file is not None:
                parse_file(file)

        except Exception as e:
            print(f'Error encountered on file upload: {e}')
            SessionStateSingleton().set_error_message('Error encountered on file upload. Is this a valid Anasys file?')
    
    st.file_uploader('Upload a file', type=['axz', 'axd'], label_visibility='collapsed',
                     on_change=on_upload, key=SessionStateSingleton().get_file_upload_widget_key())

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


def parse_map_metadata(doc):  # pylint: disable=W0613
    """
    Function to load map metadata from uploaded file.
    The file hash is used to cache the result.
    """

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

    return df

def list_heightmap_groups(doc):
    map_dict = {m.Label: m.TimeStamp for m in doc.HeightMaps.values()}
    df = pd.DataFrame([
        {
            'ts': ts,
            'Timestamp': pd.to_datetime(ts).strftime('%Y-%m-%d %H:%M:%S'),
            'Map Labels': set(m for m, v in map_dict.items() if v == ts)
        }
        for ts in set(map_dict.values())
    ]).set_index('ts').sort_index()

    return df

def parse_spectrum_metadata(doc):  # pylint: disable=W0613
    # doc = SessionStateSingleton().get_anasys_doc()

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

### PLOTTING FUNCTIONS  ---------

def plot_map(ax, hmap, zorder=None, cbar=True, sbar=True):
    """
    Plot map with applied flattening method, colormap and scalebar
    """

    # Define extent
    width = float(hmap['Size']['X'])
    height = float(hmap['Size']['Y'])
    x0 = float(hmap['Position']['X'])
    y0 = float(hmap['Position']['Y'])
    extent = (x0 - width / 2, x0 + width / 2, y0 - height / 2, y0 + height / 2)

    # Define colormap
    cmap='bwr'
    if hmap.DataChannel == 'height':
        cmap = 'afmhot'
    if 'amp' in hmap.DataChannel.lower():
        cmap = 'viridis'

    # Flatten data if necessary
    data = hmap.SampleBase64.copy()
    if hmap.DataChannel == 'deflection':
        data = data - float(hmap.Tags['Setpoint'].split(' ')[0])
    if hmap.DataChannel == 'height':
        # Subtract linear interpolation from each row
        data = data - np.array([np.linspace(row[0], row[-1], len(row)) for row in data])

    # Define colorbar limits
    vmin, vmax = np.percentile(data, .1), np.percentile(data, 99.9)
    if hmap.DataChannel == 'deflection' or hmap.DataChannel == 'phase2':
        vmax = max(vmax, -vmin)
        vmin = -vmax

    # Plot image and colorbar
    m = ax.imshow(data, extent=extent, cmap=cmap, zorder=zorder, vmin=vmin, vmax=vmax)

    if cbar:
        add_cbar(ax, m, extend='both')

    # Add scalebar
    def default_formatter(x, y):
        if y==r'$\mathrm{\mu}$m':
            y = 'µm'
        return f'{x} {y}'

    add_scalebar(ax)

    ax.set(xticks=[], yticks=[])

def add_scalebar(ax, dx=1e-6, color='w', pad=.7, scale_loc='top', loc='lower left', box_alpha=0, font_weight="heavy"):
    return ax.add_artist(ScaleBar(
        dx=dx, color=color, pad=pad, scale_loc=scale_loc, location=loc, 
        box_alpha=box_alpha, font_properties={'weight': font_weight}
    ))

def get_im_extent(hmap):
    try: 
        width = float(hmap.Size.X)
        height = float(hmap.Size.Y)
        X0 = float(hmap.Position.X)
        Y0 = float(hmap.Position.Y)
        return (X0 - width / 2, X0 + width / 2, Y0 - height / 2, Y0 + height / 2)
    except AttributeError:
        return (hmap.X.min(), hmap.X.max(), hmap.Y.min(), hmap.Y.max())



def add_cbar(ax, mappable, label='', divider=None, location='left', extend='neither'):
    """
    Add colorbar that is always as high as the image itself
    """

    if divider is None:
        divider = make_axes_locatable(ax)
    cax = divider.append_axes(location, size='5%', pad=0.1)
    ax.figure.colorbar(mappable, cax=cax, label=label, location=location, extend=extend)
    return cax


def plot_map_qc_panel(ax, map_trace, map_retrace):
    """
    Plot map with line profiles
    """
    # Plot map
    plot_map(ax, map_trace, cbar=False)

    # Add extra axes
    divider = make_axes_locatable(ax)
    cax = add_cbar(ax, ax.images[0], divider=divider, extend='both')
    ax_vert = divider.append_axes('right', size='20%', pad=0.1)
    ax_horo = divider.append_axes('bottom', size='20%', pad=0.1)

    # Plot line profiles
    n, m = map_trace.SampleBase64.shape
    ax_vert.plot(map_trace['SampleBase64'][m//2, :], -np.arange(n), lw=.5)
    ax_horo.plot(np.arange(m), map_trace['SampleBase64'][n//2, :], lw=.5)
    if map_retrace is not None:
        ax_vert.plot(map_retrace['SampleBase64'][m//2, :], -np.arange(n), lw=.5)
        ax_horo.plot(np.arange(m), map_retrace['SampleBase64'][n//2, :], lw=.5)

    # Set labels
    ax.set_title(f'{map_trace.Label} ({map_trace.UnitPrefix}{map_trace.Units})')
    ax_horo.set(xticks=[])
    ax_vert.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax_horo.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax_vert.tick_params(axis='x', rotation=90)
    ax_vert.set_yticks([])

    # def formatter(x, pos):
    #     # print(pos)
    #     if pos == 0: 
    #         print('test') 
    #         return 'test'
    #     return f'{x:.2f}'

    # cax.yaxis.set_major_formatter(formatter)

def plot_maps_qc_at_timestamp(timestamp, ncols = None):
    """
    Plot all maps at a given timestamp in one figure
    """

    doc = SessionStateSingleton().get_anasys_doc()

    # Get all channels at the given timestamp
    channels_at_timestamp = {m.DataChannel for m in doc.HeightMaps.values() if m.TimeStamp == timestamp}
    preferred_order = ['height', 'deflection', '//Func/Amplitude(x2_32,y2_32)', 'freq2', 'phase2']
    channels_at_timestamp = sorted(channels_at_timestamp, key=lambda x: preferred_order.index(x) if x in preferred_order else 999)

    if len(channels_at_timestamp) == 0:
        st.write('No maps found at this timestamp')
        return

    # Create a grid of axes
    nmaps = len(channels_at_timestamp) 
    ncols = ncols if ncols is not None else nmaps
    fig, ax = plt.subplots(1, ncols, figsize=(5 * ncols, 5), gridspec_kw={'wspace': 0.25}, squeeze=False)

    # Plot each map
    for i, channel in enumerate(channels_at_timestamp):
        # print(list(doc.HeightMaps.values()))

        maps_trace = [m for m in doc.HeightMaps.values() 
                      if m.TimeStamp == timestamp and 
                      m.DataChannel == channel and 
                      m.Tags['TraceRetrace'] == 'PrimaryTrace']
        maps_retrace = [m for m in doc.HeightMaps.values() 
                        if m.TimeStamp == timestamp and 
                        m.DataChannel == channel and 
                        m.Tags['TraceRetrace'] == 'PrimaryRetrace']
        
        if len(maps_trace) > 0:
            map_trace = maps_trace[0]
            map_retrace = None if len(maps_retrace) == 0 else maps_retrace[0]

        elif len(maps_retrace) > 0:
            map_trace = maps_retrace[0]
            map_retrace = None

        plot_map_qc_panel(ax[0, i], map_trace, map_retrace)

    for a in ax[0, nmaps:]:
        a.axis('off')

    return fig

def generate_mapqc_pdf(file_hash, ncols):
    """
    Generate a PDF file of the QC report, using an in_memory BytesIO buffer
    """

    doc = SessionStateSingleton().get_anasys_doc()
    timestamps = {m.TimeStamp for m in doc.HeightMaps.values()}

    with BytesIO() as f:
        with PdfPages(f) as pdf:
            for ts in timestamps:

                # First figure with map
                fig1 = plot_maps_qc_at_timestamp(file_hash, ts, ncols)
                main_ax = fig1.get_axes()[0]
                bbox = main_ax.get_position()  # Gets bbox in figure coordinates
                
                # Convert bbox to pixels
                x_pos = bbox.x0 * fig1.get_figwidth()
                y_pos = bbox.y1 * fig1.get_figheight()

                # Add figure to pdf
                pdf.savefig(fig1)

                # Check for QC issues
                map_labels = [k for k, m in doc.HeightMaps.items() if m.TimeStamp == ts]
                warnings = collect_map_qc_warnings(doc, map_labels)
                if len(warnings) > 0:
                
                    # Create second figure with matching dimensions
                    fig2 = plt.figure(figsize=(fig1.get_figwidth(), fig1.get_figheight()))
                    
                    # Add text at the aligned position
                    timestamp_str = pd.to_datetime(ts).strftime('%Y-%m-%d %H:%M:%S')
                    fig2.text(x_pos, y_pos, '\n - '.join(['Warnings:', *warnings]),
                            ha='left', va='top',
                            fontsize=12,
                            transform=fig2.dpi_scale_trans)
                    
                    # Save the figures
                    pdf.savefig(fig2)
                    plt.close(fig2)

                plt.close(fig1)

        return f.getvalue()
    
def sort_spectrum_datachannels(available_channels):
    """
    Sort data channels for plotting
    """
    preferred_order = ['IR Amplitude (mV)', 'PLL Frequency (kHz)', 'IR Phase (Deg)', 'Background']
    return sorted(available_channels, key=lambda x: preferred_order.index(x) if x in preferred_order else 999)

def plot_spectrum_qc(doc, selected_spectrum_label, channels_to_show, show_other_spectra=True, show_map=None, show_spectrum_labels=True):
    """
    Plot spectrum with all channels and localisation on a map
    """

    channels_to_show = sort_spectrum_datachannels(channels_to_show)

    # preferred_order = ['IR Amplitude (mV)', 'PLL Frequency (kHz)', 'IR Phase (Deg)', 'Background']
    # channels_to_show = sorted(channels_to_show, key=lambda x: preferred_order.index(x) if x in preferred_order else 999)

    # Set up figure
    if len(doc.HeightMaps) == 0:
        st.write('No height maps found')
        fig, ax_main = plt.subplots(1,1, figsize=(5, 3.4))
    else:
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={'width_ratios': [1,1]})
        ax_map, ax_main = ax

    # Set up spectrum axes
    spec_axes = []

    # If no channels are selected, show a blank figure
    if len(channels_to_show) == 0:
        ax_main.set(xticks=[], yticks=[])

    # If at least one channel is selected, register the main axis and set up its labels
    if len(channels_to_show) >= 1:
        new_ax = ax_main
        spec_axes.append(new_ax)
        new_ax.set_ylabel(channels_to_show[0], c='C0')
        new_ax.invert_xaxis()
        new_ax.set_xlabel('Wavenumber (cm⁻¹)')

    # For each additional channel, create a new axis
    if len(channels_to_show) >= 2:
        for i in range(0, len(channels_to_show) - 1):
            new_ax = ax_main.twinx()
            new_ax.spines['right'].set_visible(True)
            new_ax.spines['right'].set_position(('outward', 60 * i))
            new_ax.yaxis.set_label_position('right')
            new_ax.yaxis.set_ticks_position('right')
            spec_axes.append(new_ax)
            new_ax.set_ylabel(channels_to_show[i + 1], c=f'C{i + 1}')

    # Put the first axes on top and give them a transparent background
    for i, a in enumerate(spec_axes):
        a.set_zorder(-i)
        a.set_facecolor('none')

    # Desaturate colors
    def desat(color, fac_s = .6, fac_l = 1.5):
        rgb = to_rgb(color)
        h, l, s = colorsys.rgb_to_hls(*rgb)
        s *= fac_s
        l = min(1, l * fac_l)
        desaturated_rgb = colorsys.hls_to_rgb(h, l, s)
        return desaturated_rgb
    
    # Define a map of colors and their desaturated versions
    colors = [f'C{i}' for i in range(len(channels_to_show))]
    desaturated_colors = {c: desat(c) for c in colors}

    # Loop through spectra to plot
    for label, spectrum in doc.RenderedSpectra.items():
        # Skip if not the selected spectrum and not showing other spectra
        if label != selected_spectrum_label and not show_other_spectra: continue

        # Convert to rich dataset
        spectrum = export.spectrum_to_Dataset(spectrum)
        
        # Settings for plotting on map
        c = 'w'
        ec = 'C0'
        ms = 40
        marker = 'X'
        if label == selected_spectrum_label:
            ms = 81
            ec = 'w'
            marker = 'o'
            c = 'None'

        # Plot on map
        ax_map.scatter(
            spectrum['Location.X'], spectrum['Location.Y'],
            c=c, edgecolors=ec, s=ms, marker=marker
        )

        # Annotate on map
        if show_spectrum_labels:
            ax_map.annotate(
                label.replace('Spectrum ', ''), 
                xy=(spectrum['Location.X'], spectrum['Location.Y']), xycoords='data',
                xytext=(2,2), textcoords='offset points', color='w', fontweight='bold'
            )

        # Settings for plotting spectrum itself
        for i, channel in enumerate(channels_to_show):
            lw = .5
            c = desaturated_colors[f'C{i}']
            zorder = -1
            if label == selected_spectrum_label:
                lw=1
                c = f'C{i}'
                zorder = 0
            spec_axes[i].plot(spectrum.wavenumbers, spectrum[channel], lw=lw, c=c, zorder=zorder)

    # Find last map if necessary
    if show_map is not None:
        hmap = doc.HeightMaps[show_map]
    else:  # Default to most recent map before spectrum
        t_spectrum = pd.to_datetime(doc.RenderedSpectra[selected_spectrum_label]['TimeStamp'])
        hmaps = {
            pd.to_datetime(ts): imlist
            for (ts, tracedir), imlist in
                export.get_concurrent_images(list(doc.HeightMaps.values())).items()
        }
        if all(t_spectrum < t_map for t_map in hmaps):
            t_map = min(hmaps)
        else:
            t_map = max(t_map for t_map in hmaps if t_map <= t_spectrum)

        # Plot last map
        hmap = hmaps[t_map][0]
    
    # Plot map
    plot_map(ax_map, hmap, zorder=-1)

    return fig