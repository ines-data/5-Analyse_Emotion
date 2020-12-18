import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app
from layouts import layout1, layout2
import callbacks
from app import server


colors = {
    'background': '#DEE0E0',
    #'text': '#7FDBFF'
}
# 'backgroundColor': colors['background']
app.layout = html.Div(style={'padding': '50px','backgroundColor': colors['background']}, children=[
    html.H1('Analyse des Emotions', style={'textAlign': 'center', 'color':'rgba(35, 32, 37 ,1.7)', 'border':'3px double black'}),
    html.Br(),
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/' or pathname == '/AnalyseTraitementdesdonnées':
         return layout1
    elif pathname == '/ResultatsdesClassifications':
         return layout2
    else:
        return layout1

if __name__ == '__main__':
    app.run_server(debug=True)