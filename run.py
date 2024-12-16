import warnings
warnings.filterwarnings("ignore", message="n_jobs value 1 overridden")

import pandas as pd
import numpy as np
import umap.umap_ as umap
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import os

# Load data
df = pd.read_csv('interpolated_exp_data_with_current_and_times.csv')

size_of_plots = 450

# Group and prepare data as before
groups = df.groupby('Spectrum_id')

spectra_vectors = []
spectra_info = []
spectra_data = {}

for i, (spec_id, group) in enumerate(groups):
    # Extract arrays of Zreal/Zimag for the whole spectrum
    Zreal_orig = group['Zreal'].values
    Zimag_orig = group['Zimag'].values

    # Compute Zmod and angle
    Zmod = np.sqrt(Zreal_orig**2 + Zimag_orig**2)
    theta = np.arctan2(Zimag_orig, Zreal_orig)

    # Normalize Zmod to [0, 1]
    Zmod_min = np.min(Zmod)
    Zmod_max = np.max(Zmod)
    if Zmod_max == Zmod_min:
        Zmod_norm = np.zeros_like(Zmod)
    else:
        Zmod_norm = (Zmod - Zmod_min) / (Zmod_max - Zmod_min)
    
    # Convert normalized Zmod back to Zreal and Zimag
    Zreal_norm = Zmod_norm * np.cos(theta)
    Zimag_norm = Zmod_norm * np.sin(theta)

    vector = np.hstack([Zreal_norm, Zimag_norm])
    spectra_vectors.append(vector)

    file_name = group['file_name'].iloc[0]
    current = group['Current (A/cm²)'].iloc[0] if 'Current (A/cm²)' in group.columns else None
    time_h = group['Time (h)'].iloc[0] if 'Time (h)' in group.columns else None
    spectra_info.append((spec_id, file_name, current, time_h))

    # Store for Nyquist plotting
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

# UMAP
reducer = umap.UMAP(n_components=2, n_neighbors=50, min_dist=0.3)
embedding = reducer.fit_transform(spectra_vectors)

# Hover texts
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

# Color by Current
color = [info[2] if (info[2] is not None and not pd.isna(info[2])) else 0 for info in spectra_info]

# Create the initial Scatter (UMAP)
scatter_fig = go.Figure(
    data=[
        go.Scatter(
            x=embedding[:, 0],
            y=embedding[:, 1],
            mode='markers',
            marker=dict(color=color, colorscale='Viridis', showscale=True),
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>"
        )
    ]
)
scatter_fig.update_layout(title='UMAP of Spectra', width=size_of_plots, height=size_of_plots)

# Initial empty Nyquist plot
nyquist_orig_fig = go.Figure()
nyquist_orig_fig.update_layout(title='Nyquist Plot (Original)', width=size_of_plots, height=size_of_plots)

# Convert data to store for Dash (we can store them in-memory)
# Note: For larger datasets, consider more efficient storage or external sources.
# We'll store the entire spectra_data in a hidden div as JSON for easy callback access.
# But since these are NumPy arrays, we might need to convert them to lists.
for k,v in spectra_data.items():
    spectra_data[k]['Zreal_orig'] = v['Zreal_orig'].tolist()
    spectra_data[k]['Zimag_orig'] = v['Zimag_orig'].tolist()
    spectra_data[k]['Zreal_norm'] = v['Zreal_norm'].tolist()
    spectra_data[k]['Zimag_norm'] = v['Zimag_norm'].tolist()

# Initialize the Dash app
app = Dash(__name__)
app.layout = html.Div([
    html.Div([
        dcc.Graph(
            id='umap-graph',
            figure=scatter_fig,
            style={'display': 'inline-block'}
        ),
        dcc.Graph(
            id='nyquist-graph',
            figure=nyquist_orig_fig,
            style={'display': 'inline-block'}
        )
    ]),
    # Hidden div to store data
    html.Div(
        id='spectra-data-store',
        style={'display': 'none'},
        children=str(spectra_data) # a quick solution; for a robust solution use json.dumps
    )
])

@app.callback(
    Output('nyquist-graph', 'figure'),
    Input('umap-graph', 'clickData'),
    Input('spectra-data-store', 'children')
)
def update_nyquist(clickData, data_str):
    # Convert string back to dict
    # This approach uses eval for simplicity, 
    # but for safety and reliability, use json:
    import ast
    data = ast.literal_eval(data_str)

    fig = go.Figure()
    fig.update_layout(title='Nyquist Plot (Original)', width=size_of_plots, height=size_of_plots)

    if clickData is not None and 'points' in clickData:
        point = clickData['points'][0]  # first clicked point
        idx = point['pointIndex']       # index of the clicked point
        spectrum = data[idx]

        Zreal_orig = spectrum['Zreal_orig']
        Zimag_orig = spectrum['Zimag_orig']
        file_name = spectrum['file_name']

        fig.add_trace(go.Scatter(
            x=Zreal_orig, 
            y=[-z for z in Zimag_orig],
            mode='markers+lines',
            hovertemplate="Zreal: %{x}<br>Zimag: %{y}<extra></extra>"
        ))

        fig.update_layout(
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

if __name__ == '__main__':
    app.run_server(debug=True)
