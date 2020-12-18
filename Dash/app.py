import dash
import dash_bootstrap_components as dbc

# bootstrap theme
# https://bootswatch.com/lux/
external_stylesheets = [dbc.themes.FLATLY]

app = dash.Dash(external_stylesheets=external_stylesheets)

#app = dash.Dash (__ name__) 
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server
app.config.suppress_callback_exceptions = True
