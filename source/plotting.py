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
from utils import collect_map_qc_warnings, SessionState


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

@st.cache_resource
def plot_maps_qc_at_timestamp(file_hash, timestamp, ncols = None):
    """
    Plot all maps at a given timestamp in one figure
    """

    doc = SessionState().get_anasys_doc()

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

@st.cache_resource
def generate_mapqc_pdf(file_hash, ncols):
    """
    Generate a PDF file of the QC report, using an in_memory BytesIO buffer
    """

    doc = SessionState().get_anasys_doc()
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