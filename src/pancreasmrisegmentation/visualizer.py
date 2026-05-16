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
import math
from functools import lru_cache
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="config.yaml")
parser.add_argument("--exclude", type=str, default="excluded_dataset.csv")
parser.add_argument("--batch-size", type=int, default=60, help="Number of images per page")
try:
    args = parser.parse_args()
except SystemExit:
    class Args: config, exclude, batch_size = "config.yaml", "excluded_dataset.csv", 60
    args = Args()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

DATA_PATH = config["data_path"]
SEG_CONFIGS = config["segmentations"]

def hex_to_rgb(hex_col):
    hex_col = hex_col.lstrip('#')
    return tuple(int(hex_col[i:i+2], 16) / 255.0 for i in (0, 2, 4))


has_multiple_methods = len(SEG_CONFIGS) > 1
has_multiple_labels = any(
    len([k for k, v in c["labels"].items() if str(v).lower() != "background" and str(k) != "0"]) > 1 
    for c in SEG_CONFIGS
)

ENTITIES = []
for seg in SEG_CONFIGS:
    method_name = seg["name"]
    for label_val, label_name in seg["labels"].items():
        if str(label_name).lower() == "background" or str(label_val) == "0":
            continue
            
        if has_multiple_methods and has_multiple_labels:
            disp_name = f"{method_name} ({label_name})"
        elif has_multiple_methods:
            disp_name = method_name
        else:
            disp_name = label_name

        ENTITIES.append({
            "id": f"{method_name}::{label_val}",
            "display_name": disp_name,
            "method": method_name,
            "label_val": int(label_val),
            "label_name": label_name,
            "path": seg.get("path")
        })

palette = px.colors.qualitative.Alphabet
ENTITY_COLORS = {}
for i, ent in enumerate(ENTITIES):
    ENTITY_COLORS[ent["id"]] = hex_to_rgb(palette[i % len(palette)])

@lru_cache(maxsize=args.batch_size + 10)
def load_and_sample_volume(img_path, filename, num_slices=10):
    vol = nib.load(img_path).get_fdata()
    
    p2, p98 = np.percentile(vol, (2, 98))
    vol_norm = np.clip((vol - p2) / (p98 - p2 + 1e-8), 0, 1)
    
    segs = {}
    union_mask = np.zeros(vol.shape, dtype=bool)
    
    for seg_cfg in SEG_CONFIGS:
        method_name = seg_cfg["name"]
        seg_name = filename.replace("_0000.nii.gz", ".nii.gz")
        seg_path = os.path.join(seg_cfg["path"], seg_name)
        
        if os.path.exists(seg_path):
            seg_data = nib.load(seg_path).get_fdata().astype(np.uint8)
            segs[method_name] = seg_data
            
            for label_val, label_name in seg_cfg["labels"].items():
                if str(label_name).lower() != "background" and str(label_val) != "0":
                    union_mask |= (seg_data == int(label_val))
        else:
            segs[method_name] = np.zeros_like(vol, dtype=np.uint8)
            
    valid_slices = np.where(union_mask.any(axis=(0, 1)))[0]
    total_valid = len(valid_slices) 
    
    if total_valid == 0:
        mid = vol.shape[2] // 2
        selected_indices = np.arange(mid - num_slices//2, mid + num_slices//2)
    elif total_valid <= num_slices:
        selected_indices = valid_slices
    else:
        selected_indices = np.linspace(valid_slices[0], valid_slices[-1], num_slices, dtype=int)
        
    vol_sub = vol_norm[:, :, selected_indices]
    segs_sub = {name: data[:, :, selected_indices] for name, data in segs.items()}
    
    return vol_sub, segs_sub, tuple(selected_indices), total_valid

def create_plotly_figure(vol_subset, segs_subset, active_entity_ids):
    vol_z = np.transpose(vol_subset, (2, 0, 1))
    vol_z = np.rot90(vol_z, k=1, axes=(1, 2))
    
    Z, H, W = vol_z.shape
    blended_rgb = np.zeros((Z, H, W, 3), dtype=np.float32)
    
    segs_z = {}
    for name, data in segs_subset.items():
        z_data = np.transpose(data, (2, 0, 1))
        segs_z[name] = np.rot90(z_data, k=1, axes=(1, 2))
        
    for z in range(Z):
        base_img = np.stack((vol_z[z],)*3, axis=-1)
        
        for ent_id in active_entity_ids:
            ent = next((e for e in ENTITIES if e["id"] == ent_id), None)
            if not ent: continue
            
            method_name = ent["method"]
            label_val = ent["label_val"]
            
            if method_name not in segs_z: continue
            
            mask_data = segs_z[method_name][z]
            bool_mask = (mask_data == label_val)
                
            if np.any(bool_mask):
                color = ENTITY_COLORS[ent_id]
                alpha = 0.4
                base_img[bool_mask] = base_img[bool_mask] * (1 - alpha) + np.array(color) * alpha
                
        blended_rgb[z] = base_img

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
TOTAL_PAGES = max(1, math.ceil(len(image_files) / args.batch_size))


checklist_options = []
all_entity_ids = []
for ent in ENTITIES:
    ent_id = ent["id"]
    all_entity_ids.append(ent_id)
    rgb_color = ENTITY_COLORS[ent_id]
    css_color = f"rgb({int(rgb_color[0]*255)}, {int(rgb_color[1]*255)}, {int(rgb_color[2]*255)})"
    
    label_component = html.Span([
        html.Span(style={
            "backgroundColor": css_color, "width": "12px", "height": "12px", 
            "display": "inline-block", "marginRight": "4px", "marginLeft": "8px", 
            "verticalAlign": "middle", "borderRadius": "2px", "border": "1px solid #ccc"
        }),
        html.Span(ent["display_name"], style={"verticalAlign": "middle", "fontSize": "12px"})
    ])
    checklist_options.append({'label': label_component, 'value': ent_id})

app.layout = html.Div([
    dcc.Store(id='page-store', data=1), 
    
    dbc.Container([
        dbc.Row([
            dbc.Col(
                html.H2(f"Quality Assurance Viewer ({len(image_files)} Scans)"), 
                width=12, 
                style={"color": "#f8f9fa", "textAlign": "left"}
            ),
        ], className="py-3"),
        
        dcc.Loading(
            id="loading-grid",
            type="dot",
            color="#f8f9fa",
            children=[dbc.Row(id='image-grid')]
        ),

        dbc.Row([
            dbc.Col([
                dbc.ButtonGroup([
                    dbc.Button("⬅️ Previous", id="btn-prev", outline=True, color="light"),
                    dbc.Button(id="page-indicator", disabled=True, color="secondary", style={"color": "white", "fontWeight": "bold"}),
                    dbc.Button("Next ➡️", id="btn-next", outline=True, color="light"),
                ])
            ], width=12, className="d-flex justify-content-center mt-4 mb-5") 
        ])
        
    ], fluid=True)
], style={"backgroundColor": "#1c2833", "minHeight": "100vh", "paddingBottom": "20px"})



@app.callback(
    [Output('page-store', 'data'),
     Output('btn-prev', 'disabled'),
     Output('btn-next', 'disabled'),
     Output('page-indicator', 'children')],
    [Input('btn-prev', 'n_clicks'),
     Input('btn-next', 'n_clicks')],
    [State('page-store', 'data')],
    prevent_initial_call=False
)
def update_pagination(prev_clicks, next_clicks, current_page):
    trigger = ctx.triggered_id
    
    if trigger == 'btn-next' and current_page < TOTAL_PAGES:
        current_page += 1
    elif trigger == 'btn-prev' and current_page > 1:
        current_page -= 1
        
    prev_disabled = (current_page == 1)
    next_disabled = (current_page == TOTAL_PAGES)
    indicator_text = f"Page {current_page} of {TOTAL_PAGES}"
    
    return current_page, prev_disabled, next_disabled, indicator_text

@app.callback(
    Output('image-grid', 'children'),
    Input('page-store', 'data')
)
def render_page_grid(current_page):
    start_idx = (current_page - 1) * args.batch_size
    end_idx = start_idx + args.batch_size
    batch_files = image_files[start_idx:end_idx]
    
    excluded_set = set()
    if os.path.exists(args.exclude):
        excluded_df = pd.read_csv(args.exclude)
        excluded_set = set(excluded_df['filename'].values)
        
    cards = []
    for filename in batch_files:
        img_path = os.path.join(DATA_PATH, filename)
        
        is_excluded = filename in excluded_set
        card_style = {'opacity': '0.4', 'transition': 'opacity 0.3s'} if is_excluded else {'opacity': '1.0', 'transition': 'opacity 0.3s'}
        btn_text = "↩️ Include" if is_excluded else "❌ Exclude"
        btn_color = "success" if is_excluded else "danger"
        
        try:
            vol_sub, segs_sub, z_indices, total_valid = load_and_sample_volume(img_path, filename) 
            fig = create_plotly_figure(vol_sub, segs_sub, all_entity_ids)
            
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
                                value=all_entity_ids, 
                                inline=True,
                                labelStyle={"display": "inline-flex", "alignItems": "center", "cursor": "pointer", "marginRight": "10px", "marginBottom": "5px"}
                            ), className="text-center mb-1", style={"display": "flex", "flexWrap": "wrap", "justifyContent": "center"}
                        ),
                        dcc.Graph(
                            id={'type': 'graph', 'index': filename}, 
                            figure=fig, config={'displayModeBar': False}, style={"height": "350px"}
                        )
                    ], className="p-2"),
                    dbc.CardFooter(
                        f"Showing slices: {z_indices[0]} to {z_indices[-1]} | Total masked slices: {total_valid}", 
                        style={"fontSize": "11px", "padding": "5px 10px", "textAlign": "center"}
                    )
                ], id={'type': 'card', 'index': filename}, className="mb-4 shadow-sm", style=card_style)
            ], width=6) 
            
            cards.append(card)
        except Exception as e:
            print(f"Failed to load {filename}: {e}")
            
    return cards


@app.callback(
    Output({'type': 'graph', 'index': MATCH}, 'figure'),
    Input({'type': 'local-seg-toggle', 'index': MATCH}, 'value'),
    State({'type': 'graph', 'index': MATCH}, 'id'),
    prevent_initial_call=True
)
def update_single_graph(active_entity_ids, graph_id):
    filename = graph_id['index']
    img_path = os.path.join(DATA_PATH, filename)
    
    vol_sub, segs_sub, _, _ = load_and_sample_volume(img_path, filename) 
    
    return create_plotly_figure(vol_sub, segs_sub, active_entity_ids)

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