"""
This page shows an overview of the maps in the uploaded document.
"""

import streamlit as st
import matplotlib.pyplot as plt

import utils
from plotting import plot_map  # pylint: disable=E0401

# Set up sidebar (navigation and uploader to change file)
utils.setup_page_with_redirect(allowed_file_types=['axz', 'axd'])

st.title('Map overview')
st.write(
    '''
    This page shows an overview of the maps in the uploaded document. By default, 
    only IR Amplitude maps in the trace direction are shown, but more maps can be 
    selected below. Click on the checkmark next to each table row to select that map for plotting.
    '''
)

# Parse heightmap metadata
doc, df_heightmap_metadata = utils.parse_map_metadata(st.session_state.file_hash)

# Main content
if len(doc.HeightMaps) == 0:
    st.error('No maps found. Choose another file', icon=':material/error:')
    st.stop()

# Display maps in the placeholder, while showing a spinner until the maps are loaded.
with st.spinner('Loading maps...'):
    figure_placeholder = st.empty()


    # Display visualisation settings
    st.write('Map metadata. Use the widget above the table to select which columns to display.')

    # Choose map properties to display
    selected_metadata_tags = st.multiselect(
        label='Select properties to display',
        options=df_heightmap_metadata.columns,
        default=['Label', 'Timestamp', 'IRWavenumber', 'ScanRate'],
        label_visibility='collapsed'
    )

    # Display map properties (selectable by row)
    metadata_table = st.dataframe(
        df_heightmap_metadata, selection_mode='multi-row', on_select='rerun', hide_index=True,
        column_order=selected_metadata_tags,
        use_container_width=True
    )

    # Extract map selection
    selected_map_indices = metadata_table['selection']['rows']
    selected_map_labels = df_heightmap_metadata.iloc[selected_map_indices]['Label'].tolist()

    # If no maps are selected, take the IR maps in trace direction
    if len(selected_map_labels) == 0:
        selected_map_labels = (
            df_heightmap_metadata
            .query('DataChannel.str.contains("Ampl") and TraceRetrace == "PrimaryTrace"')
            .Label
            .tolist()
        )

    # If no IR maps in trace direction, take the first map
    if len(selected_map_labels) == 0:
        selected_map_labels = [df_heightmap_metadata.Labels[0]]

    # Check if creating figure is necessary
    if 'map_overview_fig' not in st.session_state or \
        selected_map_labels != st.session_state.selected_map_labels:

        # Store selected map labels
        st.session_state.selected_map_labels = selected_map_labels

        # Create figure
        nrow = (len(selected_map_labels) - 1) // 3 + 1
        ncol = 3  # pylint: disable=C0103
        fig, ax = plt.subplots(nrow, ncol, figsize=(5*ncol, 4.5*nrow), squeeze=False)
        ax = ax.flatten()

        # Plot each map
        i = 0  # this makes sure i is defined outside the loop
        for i, label in enumerate(selected_map_labels):
            hmap = doc.HeightMaps[label]
            plot_map(ax[i], hmap)  # TODO: annotate by wavenumber?
            ax[i].set_title(label)
            ax[i].set(xticks=[], yticks=[])

        # Hide remaining axes
        for a in ax[i+1:]:
            a.axis('off')

        st.session_state.map_overview_fig = fig

    # Display figure
    figure_placeholder.pyplot(st.session_state.map_overview_fig)