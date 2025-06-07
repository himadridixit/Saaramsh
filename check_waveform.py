import json
import numpy as np
from collections import Counter
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate


# Load waveform and spike data
with open("output/waveform.json", "r") as f:
    waveform_data = json.load(f)
    y_ds = np.array(waveform_data["waveform"])
    time_ds = np.array(waveform_data["time"])

with open("output/spikes.json", "r") as f:
    spike_times = np.array(json.load(f)["spike_times"])

# Collapse spikes within 2 seconds of each other and keep the earliest in each group
def collapse_spikes(spike_times, window=2.0):
    if len(spike_times) == 0:
        return np.array([])

    sorted_spikes = np.sort(spike_times)
    collapsed = [sorted_spikes[0]]
    
    for t in sorted_spikes[1:]:
        if t - collapsed[-1] > window:
            collapsed.append(t)
    
    return np.array(collapsed)

# Use the collapsed spike times
collapsed_spike_times = collapse_spikes(spike_times, window=2.0)
with open("output/collapsed_spikes.json", "w") as f:
    json.dump({"collapsed_spikes": collapsed_spike_times.tolist()}, f)


# Dash app
app = dash.Dash(__name__)
app.title = "Reel Inspector (cropped)"

app.layout = html.Div([
    html.H2("🎞️ Reel Inspector (cropped)"),

    # 1) Outer "crop" div: fixed width/height, overflow hidden
    html.Div(
        style={
            "width": "360px",         # desired displayed width
            "height": "640px",        # keep same aspect ratio as source (9:16)
            "overflow": "hidden",
            "margin": "auto"
        },
        children=[
            html.Video(
                controls=True,
                src="assets/VID_20250517_184414279.mp4",
                style={
                    "width": "auto",     # let the video keep its native height
                    "height": "100%",    # force video to fill the container vertically
                    # shift it left by 50% of the width to center‐crop horizontally
                    "position": "relative",
                    # "bottom": "-50px",     # adjust as needed to pick which slice you want
                }
            )
        ]
    ),

    dcc.Store(id='click-store', data={'last_click': None}),
    html.Div(id="video-timestamp", style={"display": "none"}),

    dcc.Graph(
        id="waveform",
        figure={
            "data": [
                {"x": time_ds, "y": y_ds, "mode": "lines", "name": "Waveform", "line": {"width": 1, "color": "blue"}},
                {"x": collapsed_spike_times, "y": [0.6] * len(collapsed_spike_times), "mode": "markers", "name": "Spikes", "marker": {"color": "red", "size": 4}},
            ],
            "layout": {
                "xaxis": {"title": "Time (s)"},
                "yaxis": {"title": "Amplitude"},
                "height": 300,
                "margin": {"l": 40, "r": 10, "t": 30, "b": 40},
                "hovermode": "closest",
            }
        },
        config={"displayModeBar": False}
    ),

    html.H4("📋 Collapsed Spike Times (rounded)"),
    html.Ul([html.Li(f"{t:.2f} s") for t in collapsed_spike_times[:20]])
])

click_counter = 0  # global variable

@app.callback(
    Output("video-timestamp", "children"),
    Input("waveform", "clickData")
)
def on_waveform_click(clickData):
    global click_counter

    if not clickData or "points" not in clickData:
        raise PreventUpdate

    point = clickData["points"][0]
    timestamp = float(point["x"])

    click_counter += 1

    # Output valid JSON string with counter
    return json.dumps({"time": timestamp, "count": click_counter})



if __name__ == "__main__":
    app.run(debug=True)
