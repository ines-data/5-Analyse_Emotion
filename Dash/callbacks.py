from dash.dependencies import Input, Output

from app import app

@app.callback(
    Output('app-1-display-value', 'children'),
    Input('id1', 'value'))
def display_value(value):
    return 'You have selected "{}"'.format(value)

@app.callback(
    Output('app-2-display-value', 'children'),
    Input('plot4', 'value'))
def display_value(value):
    return 'You have selected "{}"'.format(value)