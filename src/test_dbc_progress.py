import dash
from dash import html, dcc, Input, Output, callback, State, no_update
import dash_bootstrap_components as dbc
import waitress

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    dcc.Interval("progress-interval", interval=300),
    dbc.Progress(
        id="progress-bar",
        value=0,
        class_name="mb-3",
        striped=True,
        animated=True,
    ),
    dcc.Slider(0, 100, value=0, id="slider", step=1),
])


@callback(Output("progress-bar", "value"),Input("progress-interval", "n_intervals"), State("progress-bar", "value"))
def update_progress(n, value):
    return value + 1/3 if value < 100 else no_update


if __name__ == "__main__":
    waitress.serve(app.server, host="localhost", port="8050", expose_tracebacks=True, threads=8)
