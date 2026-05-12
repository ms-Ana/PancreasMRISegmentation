import dash
from dash import dcc, html, Input, Output, State, MATCH, ctx
import dash_bootstrap_components as dbc
import plotly.express as px
import numpy as np
import nibabel as nib
import yaml
import os
import argparse
import pandas as pd
from skimage.color import label2rgb
from functools import lru_cache

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config.yaml")
parser.add_argument("--exclude", type=str, default="excluded_dataset.csv")
try:
    args = parser.parse_args()
except SystemExit:
    class Args: config, exclude = "config.yaml", "excluded_dataset.csv"
    args = Args()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

DATA_PATH = config["data_path"]
SEG_CONFIG = config["segmentations"][0] 
COLORS = {"1": (1, 0, 0), "2": (0, 1, 0), "3": (0, 0, 1), "4": (1, 1, 0)}

@lru_cache(maxsize=100)
def load_and_sample_volume(img_path, seg_path, num_slices=10):
    vol = nib.load(img_path).get_fdata()
    seg = nib.load(seg_path).get_fdata().astype(np.uint8)
    
    p2, p98 = np.percentile(vol, (2, 98))
    vol_norm = np.clip((vol - p2) / (p98 - p2 + 1e-8), 0, 1)
    
    valid_slices = np.where(seg.any(axis=(0, 1)))[0]
    
    if len(valid_slices) == 0:
        mid = vol.shape[2] // 2
        selected_indices = np.arange(mid - num_slices//2, mid + num_slices//2)
    elif len(valid_slices) <= num_slices:
        selected_indices = valid_slices
    else:
        selected_indices = np.linspace(valid_slices[0], valid_slices[-1], num_slices, dtype=int)
        
    return vol_norm[:, :, selected_indices], seg[:, :, selected_indices], selected_indices

def create_plotly_figure(vol_subset, seg_subset, active_labels):
    vol_z = np.transpose(vol_subset, (2, 0, 1))
    seg_z = np.transpose(seg_subset, (2, 0, 1))
    
    vol_z = np.rot90(vol_z, k=1, axes=(1, 2))
    seg_z = np.rot90(seg_z, k=1, axes=(1, 2))
    
    Z, H, W = vol_z.shape
    blended_rgb = np.zeros((Z, H, W, 3), dtype=np.float32)
    
    for z in range(Z):
        mask = np.zeros_like(seg_z[z])
        for val in active_labels:
            if int(val) in seg_z[z]:
                mask[seg_z[z] == int(val)] = int(val)
                
        if np.any(mask):
            unique_labels = np.unique(mask)[1:] 
            slice_colors = [COLORS[str(l)] for l in unique_labels]
            blended_rgb[z] = label2rgb(mask, image=vol_z[z], colors=slice_colors, alpha=0.4, bg_label=0)
        else:
            blended_rgb[z] = np.stack((vol_z[z],)*3, axis=-1)

    fig = px.imshow(blended_rgb, animation_frame=0)
    
    fig.update_xaxes(visible=False, range=[0, W], autorange=False)
    fig.update_yaxes(visible=False, range=[H, 0], autorange=False)
    
    fig["layout"].pop("updatemenus", None)
    if fig.layout.sliders:
        fig.layout.sliders[0].pad.t = 0
        fig.layout.sliders[0].y = 0.05 
        fig.layout.sliders[0].currentvalue.visible = False

    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0), 
        coloraxis_showscale=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

image_files = sorted([f for f in os.listdir(DATA_PATH) if f.endswith(('.nii', '.nii.gz'))])

excluded_set = set()
if os.path.exists(args.exclude):
    excluded_df = pd.read_csv(args.exclude)
    excluded_set = set(excluded_df['filename'].values)

checklist_options = []
for val, name in SEG_CONFIG['labels'].items():
    rgb_color = COLORS[str(val)]
    css_color = f"rgb({int(rgb_color[0]*255)}, {int(rgb_color[1]*255)}, {int(rgb_color[2]*255)})"
    label_component = html.Span([
        html.Span(style={
            "backgroundColor": css_color, "width": "12px", "height": "12px", 
            "display": "inline-block", "marginRight": "4px", "marginLeft": "8px", 
            "verticalAlign": "middle", "borderRadius": "2px", "border": "1px solid #ccc"
        }),
        html.Span(name, style={"verticalAlign": "middle", "fontSize": "12px"})
    ])
    checklist_options.append({'label': label_component, 'value': str(val)})

initial_cards = []
for filename in image_files:
    img_path = os.path.join(DATA_PATH, filename)
    seg_path = os.path.join(SEG_CONFIG["path"], filename.replace("_0000.nii.gz", ".nii.gz"))
    
    is_excluded = filename in excluded_set
    card_style = {'opacity': '0.4', 'transition': 'opacity 0.3s'} if is_excluded else {'opacity': '1.0', 'transition': 'opacity 0.3s'}
    btn_text = "↩️ Include" if is_excluded else "❌ Exclude"
    btn_color = "success" if is_excluded else "danger"
    
    try:
        vol_sub, seg_sub, z_indices = load_and_sample_volume(img_path, seg_path)
        all_labels = [str(k) for k in SEG_CONFIG['labels'].keys()]
        fig = create_plotly_figure(vol_sub, seg_sub, all_labels)
        
        card = dbc.Col([
            dbc.Card([
                dbc.CardHeader(
                    html.Div([
                        html.Span(filename[:25] + "..." if len(filename)>25 else filename, style={"fontWeight": "bold"}),
                        dbc.Button(btn_text, id={'type': 'btn-exclude', 'index': filename}, color=btn_color, size="sm", className="float-end")
                    ])
                ),
                dbc.CardBody([
                    html.Div(
                        dcc.Checklist(
                            id={'type': 'local-seg-toggle', 'index': filename},
                            options=checklist_options,
                            value=all_labels,
                            inline=True,
                            labelStyle={"display": "inline-flex", "alignItems": "center", "cursor": "pointer"}
                        ), className="text-center mb-1"
                    ),
                    dcc.Graph(
                        id={'type': 'graph', 'index': filename}, 
                        figure=fig, config={'displayModeBar': False}, style={"height": "350px"}
                    )
                ], className="p-2"),
                dbc.CardFooter(f"Showing Z-slices: {z_indices[0]} to {z_indices[-1]}", style={"fontSize": "10px", "padding": "5px 10px"})
            ], id={'type': 'card', 'index': filename}, className="mb-4 shadow-sm", style=card_style)
        ], width=4) 
        
        initial_cards.append(card)
    except Exception as e:
        print(f"Failed to load {filename}: {e}")

app.layout = html.Div([
    dbc.Container([
        dbc.Row(dbc.Col(html.H2("Quality Assurance Viewer"), width=12, className="py-3", style={"color": "#f8f9fa", "textAlign": "center"})),
        dbc.Row(initial_cards, id='image-grid')
    ], fluid=True)
], style={"backgroundColor": "#1c2833", "minHeight": "100vh", "paddingBottom": "20px"})

@app.callback(
    Output({'type': 'graph', 'index': MATCH}, 'figure'),
    Input({'type': 'local-seg-toggle', 'index': MATCH}, 'value'),
    State({'type': 'graph', 'index': MATCH}, 'id'),
    prevent_initial_call=True
)
def update_single_graph(active_labels, graph_id):
    filename = graph_id['index']
    img_path = os.path.join(DATA_PATH, filename)
    seg_path = os.path.join(SEG_CONFIG["path"], filename.replace("_0000.nii.gz", ".nii.gz"))
    vol_sub, seg_sub, _ = load_and_sample_volume(img_path, seg_path)
    return create_plotly_figure(vol_sub, seg_sub, active_labels)

@app.callback(
    [Output({'type': 'card', 'index': MATCH}, 'style'),
     Output({'type': 'btn-exclude', 'index': MATCH}, 'children'),
     Output({'type': 'btn-exclude', 'index': MATCH}, 'color')],
    Input({'type': 'btn-exclude', 'index': MATCH}, 'n_clicks'),
    State({'type': 'card', 'index': MATCH}, 'style'),
    prevent_initial_call=True
)
def handle_exclude_toggle(n_clicks, current_style):
    if n_clicks is None: 
        return dash.no_update, dash.no_update, dash.no_update
        
    filename = ctx.triggered_id['index']
    exclude_path = args.exclude
    
    if os.path.exists(exclude_path):
        df = pd.read_csv(exclude_path)
    else:
        df = pd.DataFrame(columns=["filename"])
        
    if filename in df['filename'].values:
        df = df[df['filename'] != filename]
        df.to_csv(exclude_path, index=False)
        current_style = current_style or {}
        current_style['opacity'] = '1.0'
        return current_style, "❌ Exclude", "danger"
    else:
        df.loc[len(df)] = [filename]
        df.to_csv(exclude_path, index=False)
        current_style = current_style or {}
        current_style['opacity'] = '0.4'
        return current_style, "↩️ Include", "success"

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8050)