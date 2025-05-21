"""
This page shows an overview of the optical images in the uploaded document.
"""


import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

import utils
from utils import SessionStateSingleton

# Set up sidebar (navigation and uploader to change file)
utils.setup_page_with_redirect(allowed_file_types=['axz', 'axd'], streamlit_layout='centered')

st.title('Optical image overview')

doc = SessionStateSingleton().get_anasys_doc()

if len(doc.Images) == 0:
    st.error('No optical images found. Choose another file')
    st.stop()

# Display visualisation settings
image_key_selected = st.multiselect(
    label='Select optical image to display',
    options=list(doc.Images.keys()),
    default=list(doc.Images.keys())[:1],
)

gamma = st.slider('Gamma correction (brightness and contrast)', min_value=0.01, max_value=2.0, value=1.0, step=0.05)
show_spectra = st.checkbox('Show spectra', value=False)

show_maps = st.checkbox('Show maps', value=True)
DEFAULT_OPTION = 'Default (all IR maps)'
map_keys_to_show = st.multiselect(
    label='Show height maps',
    options=[DEFAULT_OPTION, *doc.HeightMaps.keys()],
    default=[DEFAULT_OPTION],
    disabled=not show_maps
)

if image_key_selected == []:
    st.stop()

fig, ax = plt.subplots()
ax.set_facecolor('black')
img_xmin, img_xmax, img_ymin, img_ymax = np.inf, -np.inf, np.inf, -np.inf

for key in image_key_selected:
    # st.write(image_selected)
    image_selected = doc.Images[key]
    data = image_selected.SampleBase64
    data = data[..., 0:3]/255  # Delete the alpha channel
    data = data**gamma
    
    extent = utils.get_im_extent(image_selected)
    ax.imshow(data, extent=extent)
    img_xmin = min(img_xmin, extent[0])
    img_xmax = max(img_xmax, extent[1])
    img_ymin = min(img_ymin, extent[2])
    img_ymax = max(img_ymax, extent[3])

ax.set(xticks=[], yticks=[], xlim=[img_xmin, img_xmax], ylim=[img_ymin, img_ymax])

if show_spectra:
    ax.set(xlim=ax.get_xlim(), ylim=ax.get_ylim())
    xs = np.array([s['Location']['X'] for s in doc.RenderedSpectra.values()])
    ys = np.array([s['Location']['Y'] for s in doc.RenderedSpectra.values()])
    labels = list(doc.RenderedSpectra.keys())
    mask = (xs >= img_xmin) & (xs <= img_xmax) & (ys >= img_ymin) & (ys <= img_ymax)
    xs, ys, labels = np.array(xs)[mask], np.array(ys)[mask], np.array(labels)[mask]
    colors = plt.cm.viridis(np.linspace(0, 1, len(xs)))
    for x, y, label, color in zip(xs, ys, labels, colors):
        ax.scatter(x, y, c=color, s = 2, label=label)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize='small', title='Spectrum labels')

if show_maps:
    if DEFAULT_OPTION in map_keys_to_show:
        irmapkeys = [k for k, map in doc.HeightMaps.items()
                    if 'IR Amplitude' in k
                    or '//Func/Amplitude(x2_32,y2_32)' == map.DataChannel]

        map_keys_to_show = {k for k in [*map_keys_to_show, *irmapkeys]
                            if k != DEFAULT_OPTION}
    for k in map_keys_to_show:
        m = doc.HeightMaps[k]
        utils.plot_map(ax, m, cbar=False, sbar=False)

utils.add_scalebar(ax)
st.pyplot(fig)
