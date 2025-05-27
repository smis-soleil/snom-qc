"""
This page shows an overview of the maps in the uploaded document.
"""

import streamlit as st
import matplotlib.pyplot as plt

import utils
from utils import SessionStateSingleton

# Set up sidebar (navigation and uploader to change file)
utils.setup_page_with_redirect(allowed_file_types=['axz', 'axd'])

st.title('Map overview')
st.write(
    '''
    This page shows an overview of the maps in the uploaded document. By default, 
    only IR Amplitude maps in the trace direction are shown, but more maps can be 
    selected below. Click on the checkmark next to each table row to select that map for plotting.

    As little processing as possible is done on this data. Height maps are plotted
    after a line subtraction, and deflection maps are relative to the setpoint.
    '''
)

# Parse heightmap metadata
doc = SessionStateSingleton().get_anasys_doc()
df_heightmap_metadata = SessionStateSingleton().get_cached_heightmap_metadata()

# Main content
if len(doc.HeightMaps) == 0:
    st.error('No maps found. Choose another file', icon=':material/error:')
    st.stop()

# Display visualisation settings
with st.expander('View metadata and select maps to display', expanded=False):

    # Choose map properties to display
    st.caption('Select metadata properties')
    selected_metadata_tags = st.multiselect(
        label='Metadata properties to display',
        options=df_heightmap_metadata.columns,
        default=['Label', 'Timestamp', 'IRWavenumber', 'ScanRate'],
        label_visibility='collapsed'
    )

    # Display map properties (selectable by row)
    st.caption('Metadata table. Select row to display the corresponding map')
    metadata_table = st.dataframe(
        df_heightmap_metadata, selection_mode='multi-row', on_select='rerun', hide_index=True,
        column_order=selected_metadata_tags,
        use_container_width=True,
    )

    # Extract map selection
    selected_map_indices = metadata_table['selection']['rows']
    selected_map_labels = df_heightmap_metadata.iloc[selected_map_indices]['Label'].tolist()

    # If no maps are selected, take the IR maps in trace direction
    if len(selected_map_labels) == 0:
        selected_map_labels = (
            df_heightmap_metadata
            .query('(DataChannel.str.contains("Ampl") or DataChannel.str.contains("ampl")) and TraceRetrace == "PrimaryTrace"')
            .Label
            .tolist()
        )

    # If no IR maps in trace direction, take the first map
    if len(selected_map_labels) == 0:
        selected_map_labels = [df_heightmap_metadata.Label[0]]

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
    utils.plot_map(ax[i], hmap)
    ax[i].annotate(hmap.Tags['IRWavenumber'], (0.05, 0.95), xycoords='axes fraction', ha='left', va='top', weight='bold', c='w')
    ax[i].set_title(label)
    ax[i].set(xticks=[], yticks=[])

# Hide remaining axes
for a in ax[i+1:]:
    a.axis('off')

# Display figure
st.pyplot(fig)