import warnings
# Filter out the FutureWarning about 'force_all_finite' -> 'ensure_all_finite'
warnings.filterwarnings(
    "ignore", 
    category=FutureWarning, 
    message=".*'force_all_finite' was renamed to 'ensure_all_finite' in 1.6 and will be removed in 1.8.*"
)

import pandas as pd
import numpy as np
import umap.umap_ as umap
import plotly.graph_objects as go

from dash import Dash, html, dcc, Input, Output
import dash
import os

# Load data
# Ensure that your CSV file is included in the repository. If it's large, consider hosting it elsewhere.
data_file = 'interpolated_exp_data_with_current_and_times.csv'
df = pd.read_csv(data_file)

size_of_plots = 600

# Group by Spectrum_id
groups = df.groupby('Spectrum_id')

spectra_vectors = []
spectra_info = []
spectra_data = {}

for i, (spec_id, group) in enumerate(groups):
    # Extract arrays of Zreal/Zimag for the whole spectrum
    Zreal_orig = group['Zreal'].values
    Zimag_orig = group['Zimag'].values

    # Compute Zmod and angle for normalization
    Zmod = np.sqrt(Zreal_orig**2 + Zimag_orig**2)
    theta = np.arctan2(Zimag_orig, Zreal_orig)  # preserves sign and quadrant

    # Normalize Zmod to [0, 1]
    Zmod_min = np.min(Zmod)
    Zmod_max = np.max(Zmod)
    if Zmod_max == Zmod_min:
        # Edge case: all values are the same
        Zmod_norm = np.zeros_like(Zmod)
    else:
        Zmod_norm = (Zmod - Zmod_min) / (Zmod_max - Zmod_min)
    
    # Convert normalized Zmod back to Zreal and Zimag
    Zreal_norm = Zmod_norm * np.cos(theta)
    Zimag_norm = Zmod_norm * np.sin(theta)

    # Create a vector by stacking normalized Zreal and Zimag for dimension reduction
    vector = np.hstack([Zreal_norm, Zimag_norm])
    spectra_vectors.append(vector)
    
    # Extract representative info
    file_name = group['file_name'].iloc[0]
    current = group['Current (A/cm²)'].iloc[0] if 'Current (A/cm²)' in group.columns else None
    time_h = group['Time (h)'].iloc[0] if 'Time (h)' in group.columns else None
    spectra_info.append((spec_id, file_name, current, time_h))
    
    # Store full normalized and original data for Nyquist plotting
    spectra_data[i] = {
        'Zreal_norm': Zreal_norm,
        'Zimag_norm': Zimag_norm,
        'Zreal_orig': Zreal_orig,
        'Zimag_orig': Zimag_orig,
        'file_name': file_name,
        'Current': current,
        'Time': time_h
    }

spectra_vectors = np.array(spectra_vectors)

# Apply UMAP
reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=50, min_dist=0.3)
embedding = reducer.fit_transform(spectra_vectors)

# Generate hover texts for Current and Time
hover_texts = []
for (_, file_name, current, time_h) in spectra_info:
    hover_str = ""
    if pd.notna(current):
        hover_str += f"Current: {current} A/cm²<br>"
    else:
        hover_str += "Current: None<br>"
    if pd.notna(time_h):
        hover_str += f"Time: {time_h} h"
    else:
        hover_str += "Time: None"
    hover_texts.append(hover_str)

# Build Dash app
app = Dash(__name__)
server = app.server  # Expose the underlying Flask server for deployment

app.layout = html.Div([
    html.Div([
        html.Label("Color by:"),
        dcc.RadioItems(
            id='color-by-radio',
            options=[
                {'label': 'Current', 'value': 'current'},
                {'label': 'Time', 'value': 'time'}
            ],
            value='current',  # Default selection
            style={'margin-bottom': '20px'}
        )
    ]),
    html.Div([
        dcc.Graph(
            id='umap-scatter',
            style={'display': 'inline-block', 'vertical-align': 'top'}
        ),
        dcc.Graph(
            id='nyquist-plot',
            style={'display': 'inline-block', 'vertical-align': 'top'}
        )
    ])
])

# Callback to update the UMAP scatter based on selected coloring
@app.callback(
    Output('umap-scatter', 'figure'),
    Input('color-by-radio', 'value')
)
def update_umap_scatter(color_by):
    if color_by == 'current':
        # Color by current
        colors = [info[2] if (info[2] is not None and not pd.isna(info[2])) else 0 for info in spectra_info]
        color_title = "Current (A/cm²)"
    else:
        # Color by time
        colors = [info[3] if (info[3] is not None and not pd.isna(info[3])) else 0 for info in spectra_info]
        color_title = "Time (h)"

    scatter_fig = go.Figure(data=[
        go.Scatter(
            x=embedding[:, 0],
            y=embedding[:, 1],
            mode='markers',
            marker=dict(
                color=colors, 
                colorscale='Viridis', 
                showscale=True,
                colorbar=dict(title=color_title)
            ),
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>",
        )
    ])
    scatter_fig.update_layout(title='UMAP of Spectra', width=size_of_plots, height=size_of_plots)
    return scatter_fig

# Callback to update nyquist plot based on selected point
@app.callback(
    Output('nyquist-plot', 'figure'),
    Input('umap-scatter', 'clickData')
)
def update_nyquist(clickData):
    # If no point is clicked, return an empty figure
    if clickData is None:
        fig = go.Figure()
        fig.update_layout(title='Nyquist Plot (Original)', width=size_of_plots, height=size_of_plots)
        return fig

    # Extract the selected point index
    point_index = clickData['points'][0]['pointIndex']
    data = spectra_data[point_index]

    Zreal_orig = data['Zreal_orig']
    Zimag_orig = data['Zimag_orig']
    file_name = data['file_name']

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=Zreal_orig, 
        y=-Zimag_orig, 
        mode='markers+lines',
        hovertemplate="Zreal: %{x}<br>Zimag: %{y}<extra></extra>"
    ))
    fig.update_layout(
        title="Nyquist Plot (Original)",
        width=size_of_plots,
        height=size_of_plots,
        annotations=[
            dict(
                x=0.5,
                y=1.1,
                xref="paper",
                yref="paper",
                text=f"<b>File:</b> {file_name}",
                showarrow=False,
                font=dict(size=10)
            )
        ]
    )

    return fig
