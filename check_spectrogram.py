import json
import numpy as np
import librosa
import librosa.display
import cv2
import matplotlib.pyplot as plt
import plotly.graph_objs as go

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import scipy.signal

## 44, 45, 46

def high_pass_filter(y, sr, cutoff=300):
    b, a = scipy.signal.butter(4, cutoff / (0.5 * sr), btype='high')
    return scipy.signal.filtfilt(b, a, y)

def normalize_audio(y):
    return y / np.max(np.abs(y))

# --- Spectrogram Matching Logic ---
def load_audio(path, sr=22050):
    y, _ = librosa.load(path, sr=sr)
    return y

def audio_to_mel_spectrogram(y, sr=22050, hop_length=512, n_mels=128):
    S = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=hop_length, n_mels=n_mels)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB

def match_spectrogram(template_dB, full_dB, threshold=0.7):
    # Normalize and convert to 8-bit images
    def to_img(S):
        S = 255 * (S - S.min()) / (S.max() - S.min())
        return S.astype(np.uint8)
    
    template_img = to_img(template_dB)
    full_img = to_img(full_dB)

    result = cv2.matchTemplate(full_img, template_img, cv2.TM_CCOEFF_NORMED)
    match_locations = np.where(result >= threshold)
    return sorted(set(match_locations[1]))  # x-axis = time

# Load audio and match
sr = 44100
hop = 256
print("1")
template_audio = load_audio("output/template.wav", sr)
full_audio = load_audio("output/audio.wav", sr)
print("loaded audio")
template_dB = audio_to_mel_spectrogram(template_audio, sr, hop)
full_dB = audio_to_mel_spectrogram(full_audio, sr, hop)
print("converted to mel spectrogram")
matches_x = match_spectrogram(template_dB, full_dB, threshold=0.9)
match_times = [x * hop / sr for x in matches_x]
rounded_matches = sorted(set(np.round(match_times, 1)))
with open("output/match_times.json", "w") as f:
    json.dump({"match_times": match_times}, f)
print("matching done")  

librosa.display.specshow(template_dB, x_axis='time', y_axis='mel')
librosa.display.specshow(full_dB, x_axis='time', y_axis='mel')

# Create spectrogram figure
fig = go.Figure()

n_frames = full_dB.shape[1]
time_axis = np.arange(n_frames) * hop / sr

fig.add_trace(go.Heatmap(
    z=full_dB,
    x=time_axis,          # <- Set x explicitly
    y=np.arange(full_dB.shape[0]),  # Optional: set mel bins explicitly
    colorscale='Viridis',
    showscale=False
))

# Add red lines for matches
for t in rounded_matches:
    fig.add_shape(
        type='line',
        x0=t, x1=t,
        y0=0, y1=128,
        line=dict(color='red', width=2)
    )

# # Add vertical lines as a single scatter trace
# line_x = []
# line_y = []

# for t in rounded_matches:
#     line_x += [t, t, None]  # None separates segments
#     line_y += [0, 128, None]

# fig.add_trace(go.Scatter(
#     x=line_x,
#     y=line_y,
#     mode='lines',
#     line=dict(color='red', width=1),
#     name='Matches',
#     showlegend=False
# ))

fig.update_layout(
    title="Spectrogram with Template Matches",
    xaxis_title="Time (s)",
    yaxis_title="Mel Frequency Bin",
    margin=dict(l=40, r=40, t=40, b=40),
    height=400,
    hovermode='x'
)
# --- Dash App ---
app = dash.Dash(__name__)
app.title = "SpectroReel Inspector"

app.layout = html.Div([
    html.H2("🎙️ SpectroReel Inspector"),

    html.Video(
        controls=True,
        children=[
            html.Source(src="assets/VID_20250517_184414279.mp4", type="video/mp4")
        ],
        # src="videos/VID_20250517_184414279.mp4",
        style={"width": "80%", "display": "block", "margin": "auto"}
    ),

    dcc.Store(id='click-store', data={'last_click': None}),
    html.Div(id="video-timestamp", style={"display": "none"}),

    dcc.Graph(id="spectrogram", figure=fig, config={"displayModeBar": False}),
])
# --- Callback to handle clicks ---
@app.callback(
    Output("video-timestamp", "children"),
    Input("spectrogram", "clickData")
)
def on_click(clickData):
    if not clickData or "points" not in clickData:
        raise PreventUpdate

    point = clickData["points"][0]
    timestamp = float(point["x"])
    return json.dumps({"time": timestamp})

if __name__ == "__main__":
    app.run(debug=True)


# Switch from image matching to MFCC + DTW