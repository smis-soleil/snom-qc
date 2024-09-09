"""
Plotting utilities. Most of these should be going into the anasyspythontools package.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import pandas as pd
import streamlit as st
from anasyspythontools import export
import colorsys
from matplotlib.colors import to_rgb
from io import BytesIO
from matplotlib.backends.backend_pdf import PdfPages


def plot_map(ax, hmap, zorder=None, cbar=True):
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

    # Plot image and colorbar
    m = ax.imshow(data, extent=extent, cmap=cmap, zorder=zorder)

    if cbar:
        add_cbar(ax, m)

    # Add scalebar
    def default_formatter(x, y):
        if y==r'$\mathrm{\mu}$m':
            y = 'µm'
        return f'{x} {y}'

    ax.add_artist(ScaleBar(
        dx=1e-6,
        pad=.7,
        box_alpha=0,
        color='w',
        location='lower left',
        scale_formatter=default_formatter,
        font_properties={'weight': 'bold'}
    ))

    ax.set(xticks=[], yticks=[])


def add_cbar(ax, mappable, label='', divider=None, location='left'):
    """
    Add colorbar that is always as high as the image itself
    """

    if divider is None:
        divider = make_axes_locatable(ax)
    cax = divider.append_axes(location, size='5%', pad=0.1)
    ax.figure.colorbar(mappable, cax=cax, label=label, location=location)


def plot_map_qc(ax, map_trace, map_retrace):
    """
    Plot map with line profiles
    """

    # Plot map
    plot_map(ax, map_trace, cbar=False)

    # Add extra axes
    divider = make_axes_locatable(ax)
    add_cbar(ax, ax.images[0], divider=divider)
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
    ax.set_title(map_trace['Label'])
    ax_vert.set_yticks([])
    ax_vert.set_xticks(ax_vert.get_xticks())
    ax_vert.set_xticklabels(ax_vert.get_xticklabels(), rotation=70)
    ax_horo.set(xticks=[])

@st.cache_resource
def plot_maps_qc(file_hash, timestamp):
    """
    Plot all maps at a given timestamp in one figure
    """

    doc = st.session_state.anasys_doc

    maps_trace = [m for m in doc.HeightMaps.values()
                  if m.TimeStamp == timestamp and m.Tags['TraceRetrace'] == 'PrimaryTrace']
    maps_retrace = [m for m in doc.HeightMaps.values()
                    if m.TimeStamp == timestamp and m.Tags['TraceRetrace'] == 'PrimaryRetrace']

    # Create a grid of axes
    nmaps = len(maps_trace)
    fig, ax = plt.subplots(1, nmaps, figsize=(5 * nmaps, 5))

    # Plot each map
    for i, map_trace in enumerate(maps_trace):
        map_retrace = [m for m in maps_retrace if m['Label'] == map_trace['Label']]
        if len(map_retrace) > 0:
            map_retrace = map_retrace[0]
        else:
            map_retrace = None

        plot_map_qc(ax[i], map_trace, map_retrace)

    return fig

@st.cache_resource
def generate_mapqc_pdf(file_hash):
    """
    Generate a PDF file of the QC report, using an in_memory BytesIO buffer
    """

    timestamps = {m.TimeStamp for m in st.session_state.anasys_doc.HeightMaps.values()}

    with BytesIO() as f:
        with PdfPages(f) as pdf:
            for ts in timestamps:
                fig = plot_maps_qc(file_hash, ts)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

        return f.getvalue()


def plot_spectrum_qc(doc, selected_spectrum_label, channels_to_show, show_other_spectra=True, show_map=None, show_spectrum_labels=True):
    """
    Plot spectrum with all channels and localisation on a map
    """

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