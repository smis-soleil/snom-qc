"""
This page shows an overview of the optical images in the uploaded document.
"""


import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

import utils
import plotting  # pylint: disable=E0401

# Set up sidebar (navigation and uploader to change file)
utils.setup_page_with_redirect(allowed_file_types=['axz', 'axd'], streamlit_layout='centered')

st.title('Optical image overview')

# TODO
st.write('==check scalebar is correct==')
st.write('==choose new example file==')

doc = st.session_state.anasys_doc

if len(doc.Images) == 0:
    st.error('No optical images found. Choose another file')
    st.stop()

# Display visualisation settings
image_key_selected = st.selectbox(
    label='Select optical image to display',
    options=doc.Images.keys(),
    index=0
)
gamma = st.slider('Gamma correction', min_value=0.01, max_value=2.0, value=1.0, step=0.05)
show_spectra = st.checkbox('Show spectra', value=False)

show_maps = st.checkbox('Show maps', value=False)
DEFAULT_OPTION = 'Default (all IR maps)'
map_keys_to_show = st.multiselect(
    label='Show height maps',
    options=[DEFAULT_OPTION, *doc.HeightMaps.keys()],
    default=[DEFAULT_OPTION],
    disabled=not show_maps
)

image_selected = doc.Images[image_key_selected]
data = image_selected.SampleBase64
data = data[..., 0:3]/255  # Delete the alpha channel
data = data**gamma

fig, ax = plt.subplots()
ax.imshow(data, extent=plotting.get_im_extent(image_selected))
image_limits = ax.get_xlim(), ax.get_ylim()
ax.set(xticks=[], yticks=[])

if show_spectra:
    ax.set(xlim=ax.get_xlim(), ylim=ax.get_ylim())
    xs = [s['Location']['X'] for s in doc.RenderedSpectra.values()]
    ys = [s['Location']['Y'] for s in doc.RenderedSpectra.values()]
    labels = list(doc.RenderedSpectra.keys())
    mask = (xs >= ax.get_xlim()[0]) & (xs <= ax.get_xlim()[1]) & (ys >= ax.get_ylim()[0]) & (ys <= ax.get_ylim()[1])
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
        plotting.plot_map(ax, m, cbar=False, sbar=False)
        # ax.imshow(m.SampleBase64,
        #           extent=plotting.get_im_extent(m), 
        #           alpha=1, vmax = np.quantile(m.SampleBase64, .95))

ax.set(xlim = image_limits[0], ylim = image_limits[1])

plotting.add_scalebar(ax)
st.pyplot(fig)
