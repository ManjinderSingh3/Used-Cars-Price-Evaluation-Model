import dash
import dash_table
import numpy

from dash.dependencies import Input, Output, State

import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from graphs import *
import pandas as pd
import plotly.express as px
from prediction_models import *

#  References:
#  https://plotly.com/python/
#  https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
#  https://dash.plotly.com/layout
#  https://plotly.com/python/pie-charts/
#  https://plotly.com/python/builtin-colorscales/

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server


models = ["Random Forest Regressor"]
vehicle_df = pd.read_csv(Path('Data', 'vehicle_final.csv'))
default_display_columns = ['manufacturer', 'model', 'transmission', 'odometer', 'year', 'price']
row_count = np.count_nonzero(vehicle_df.index)


card_year = dbc.Card([
    dbc.CardBody(
        [
            dbc.Label(['Choose Manufacturing Year'], style={'font-weight': 'bold', "text-align": "center"}),
            dcc.Dropdown(id="year_dropdown",
                         value = vehicle_df['year'].unique()[0],
                         options=[{"label": i, "value": i} for i in sorted(vehicle_df['year'].unique())],
                         searchable=True,  # This parameter helps user to search from dropdown
                         placeholder='Please select...',  # Default text when no option is selected
                         clearable=True,  # User can remove selected value from dropdown
                         style={'width': "90%"},  # use dictionary to define CSS styles of your dropdown
                         )
        ]),
])

card_manufacturer = dbc.Card([
    dbc.CardBody(
        [
            dbc.Label(['Choose Manufacturer'], style={'font-weight': 'bold', "text-align": "center"}),
            dcc.Dropdown(id="manufacturer_dropdown",
                         options=[{"label": i, "value": i} for i in vehicle_df['manufacturer'].unique()],
                         value=vehicle_df['manufacturer'].unique()[0],
                         searchable=True,
                         )
        ]),
])

card_model = dbc.Card([
    dbc.CardBody(
        [
            dbc.Label(['Choose Model'], style={'font-weight': 'bold', "text-align": "center"}),
            dcc.Dropdown(id="model_dropdown",
                         value=vehicle_df['model'].unique()[0],
                         searchable=True,
                         )
        ]),
])


card_fuel_type = dbc.Card([
    dbc.CardBody(
        [
            dbc.Label(['Select Fuel Type'], style={'font-weight': 'bold', "text-align": "center"}),
            dcc.Dropdown(id="fuel_type_dropdown",
                         value=vehicle_df['fuel'].unique()[0],
                         options=[{"label": i, "value": i} for i in vehicle_df['fuel'].unique()],
                         searchable=True,  # This parameter helps user to search from dropdown
                         placeholder='Please select...',  # Default text when no option is selected
                         clearable=True,  # User can remove selected value from dropdown
                         style={'width': "100%"},  # use dictionary to define CSS styles of your dropdown
                         )
        ]),
])

card_odometer_reading = dbc.Card([
    dbc.CardBody(
        [
            dbc.Label(['Select Odometer reading'], style={'font-weight': 'bold', "text-align": "center"}),
            dcc.Input(id="odometer_reading", type="number",
                      value=68696,
                      style={'width': '100%'},
                      placeholder='Please select...')
        ]),
])

card_vehicle_status = dbc.Card([
    dbc.CardBody(
        [
            dbc.Label(['Choose status of vehicle'], style={'font-weight': 'bold', "text-align": "center"}),
            dcc.Dropdown(id="vehicle_status_dropdown",
                         value=vehicle_df['title_status'].unique()[0],
                         options=[{"label": i, "value": i} for i in vehicle_df['title_status'].unique()],
                         searchable=True,  # This parameter helps user to search from dropdown
                         placeholder='Please select...',  # Default text when no option is selected
                         clearable=True,  # User can remove selected value from dropdown
                         style={'width': "100%"},
                         )
        ]),
])

card_transmission_type = dbc.Card([
    dbc.CardBody(
        [
            dbc.Label(['Select Transmission Type'], style={'font-weight': 'bold', "text-align": "center"}),
            dcc.Dropdown(id="transmission_dropdown",
                         value=vehicle_df['transmission'].unique()[0],
                         options=[{"label": i, "value": i} for i in vehicle_df['transmission'].unique()],
                         searchable=True,  # This parameter helps user to search from dropdown
                         placeholder='Please select...',  # Default text when no option is selected
                         clearable=True,  # User can remove selected value from dropdown
                         style={'width': "100%"},
                         )
        ]),
])

card_drivetrain = dbc.Card([
    dbc.CardBody(
        [
            dbc.Label(['Select Train Type'], style={'font-weight': 'bold', "text-align": "center"}),
            dcc.Dropdown(id="traintype_dropdown",
                         value=vehicle_df['drive'].unique()[0],
                         options=[{"label": i, "value": i} for i in vehicle_df['drive'].unique()],
                         searchable=True,  # This parameter helps user to search from dropdown
                         placeholder='Please select...',  # Default text when no option is selected
                         clearable=True,  # User can remove selected value from dropdown
                         style={'width': "80%"},
                         )
        ]),
])

card_vehicle_type = dbc.Card([
    dbc.CardBody(
        [
            dbc.Label(['Select Vehicle Type'], style={'font-weight': 'bold', "text-align": "center"}),
            dcc.Dropdown(id="vehicle_type_dropdown",
                         value=vehicle_df['type'].unique()[0],
                         options=[{"label": i, "value": i} for i in vehicle_df['type'].unique()],
                         searchable=True,  # This parameter helps user to search from dropdown
                         placeholder='Please select...',  # Default text when no option is selected
                         clearable=True,  # User can remove selected value from dropdown
                         style={'width': "80%"},
                         )
        ]),
])

card_paint_color = dbc.Card([
    dbc.CardBody(
        [
            dbc.Label(['Select Paint Color'], style={'font-weight': 'bold', "text-align": "center"}),
            dcc.Dropdown(id="paint_dropdown",
                         value=vehicle_df['paint_color'].unique()[0],
                         options=[{"label": i, "value": i} for i in vehicle_df['paint_color'].unique()],
                         searchable=True,  # This parameter helps user to search from dropdown
                         placeholder='Please select...',  # Default text when no option is selected
                         clearable=True,  # User can remove selected value from dropdown
                         style={'width': "80%"},
                         )
        ]),
])

card_model_selection = dbc.Card([
    dbc.CardBody(
        [
            dbc.Label(['Model used'], style={'font-weight': 'bold', "text-align": "center"}),
            dcc.Dropdown(id="ml_models_dropdown", disabled=True,
                         value=models[0],
                         options=[{"label": i, "value": i} for i in models],
                         searchable=True,  # This parameter helps user to search from dropdown
                         placeholder='Please select...',  # Default text when no option is selected
                         clearable=True,  # User can remove selected value from dropdown
                         style={'width': "90%"},
                         ),
            html.Br(),
            dbc.Button('Predict Price', id='predict_price', color='primary', style={'margin-bottom': '1em'},
                       block=True),
        ]),
])

card_predicted_price = dbc.Card([
    dbc.CardBody(
        [
            html.H5("Predicted price in USD is : ", className="card-title"),
            html.P(" ", id="predicted_price", className="card-text")
        ]),
])

card_upper_range = dbc.Card([
    dbc.CardBody(
        [
            html.H2("Suggested Upper Limit", className="card-title"),
            html.H5("Note: This price is 10% above the Predicted Price.", className="card-title"),
            html.P(" ", id="upper_range", className="card-text")
        ]),
])

card_lower_range = dbc.Card([
    dbc.CardBody(
        [
            html.H2("Suggested Lower Limit", className="card-title"),
            html.H5("Note: This price is 10% below the Predicted Price.", className="card-title"),
            html.P(" ", id="lower_range", className="card-text")
        ]),
])

card_feature_importance = dbc.Card([
    dbc.CardBody(
        [
            dcc.Graph(id='feature_importance', figure={}),
        ]),
])


before_label_encoding = read_csv_file(Path('Data', 'vehicle_final.csv'))
after_label_encoding = read_csv_file(Path('Data', 'vehicle_final_le.csv'))
sorted_vehicle_df = before_label_encoding.sort_values(by=['year'])

animations = {
    'Scatter': px.scatter(
        sorted_vehicle_df, x="odometer", y="price", animation_frame='year',
        animation_group="model", size="year", color="manufacturer",
        hover_name="model", log_x=True, size_max=5,
        range_x=[1000, 100000], range_y=[10000, 60000]),
}

app.layout = dbc.Container([
    html.H1("Used Car Price Analytics", style={"margin-bottom": '1em', "text-align": "center"}),
    dcc.Tabs([
        dcc.Tab(label="View Dataset", children=[
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        dbc.FormGroup([
                            html.Br(),
                            html.Label("Select Fields"),
                            dcc.Dropdown(
                                id="fields_dropdown",
                                options=[{"label": i, "value": i} for i in vehicle_df.columns],
                                value=default_display_columns,
                                multi=True
                            )
                        ])
                    ])
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.FormGroup([
                            html.Label("Records per page"),
                            dcc.Slider(
                                id="records_on_single_page",
                                min=0,
                                max=50,
                                value=10,
                                step=10,
                                marks={
                                    0: '0',
                                    10: '10',
                                    20: '20',
                                    30: '30',
                                    40: '40',
                                    50: '50'
                                }
                            )
                        ])
                    ])
                ]),
                dbc.Button('Show Data', id='display_data', color='primary', style={"text-align": "center"}),
                html.Hr(),
                dbc.Row([
                    dbc.Col([
                        dash_table.DataTable(
                            id="dataframe_table",
                            editable=True,
                            filter_action="native",
                            sort_mode="multi",
                            page_action="native",
                            sort_action="native",
                            page_current=0,
                            page_size=10,
                            style_cell={'textAlign': 'left'},
                            style_header={
                                'backgroundColor': 'grey',
                                'fontWeight': 'bold',
                                'font': 'upper'
                            }
                        )
                    ])
                ])
            ])
        ]),
        dcc.Tab(label="Data Visualization", children=[
            dbc.Container([
                dbc.Row([
                    dbc.Col([
                        dbc.FormGroup([
                            dbc.Label("Select the visualization"),
                            dcc.Dropdown(id="visualization_type", value="avg_pr",
                                         options=[{"label": "Available cars based on color",
                                                   "value": "available_color"},
                                                  {"label": "Available vehicles based on type",
                                                   "value": "available_type"},
                                                  {"label": "Average price based on fuel type", "value": "avg_pr"},
                                                  {"label": "Chart showing number of cars available per year",
                                                   "value": "cars_available_per_year"},
                                                  {"label": "Correlation heatmap", "value": "correlation_heatmap"},
                                                  {"label": "Histogram showing number of cars at different price",
                                                   "value": "histogram"},
                                                  {"label": "Pie chart showing available models per manufacturer",
                                                   "value": "pie_chart"}
                                                  ])
                        ]),
                        dbc.FormGroup([
                            dbc.Button("Show Visualization", id="visualization_button", color='primary')
                        ])
                    ])
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.FormGroup([
                            dcc.Graph(id='visual_graph')
                        ])
                    ])
                ]),
                dbc.Row([
                    html.H1("Select columns to see different visualizations"),
                ]),
                html.Hr(),
                dbc.Row([
                    dbc.Col([
                        dbc.FormGroup([
                            dbc.Label("Select column for X-axis"),
                            dcc.Dropdown(id="x_axis", value="year",
                                         options=[{"label": i, "value": i} for i in vehicle_df.columns])
                        ])
                    ]),
                    dbc.Col([
                        dbc.FormGroup([
                            dbc.Label("Select column for Y-axis"),
                            dcc.Dropdown(id="y_axis", value="price",
                                         options=[{"label": i, "value": i} for i in vehicle_df.columns])
                        ])
                    ])
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.FormGroup([
                            dbc.Label("Select column to filter the data"),
                            dcc.Dropdown(id='filter_column', value="None", clearable=False,
                                         options=[{"label": i, "value": i} for i in vehicle_df.columns])
                        ])
                    ]),
                    dbc.Col([
                        dbc.FormGroup([
                            dbc.Label("Select column value to filter the data"),
                            dcc.Dropdown(id='select_value', clearable=False)
                        ])
                    ])
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.FormGroup([
                            dbc.Label("Select the Graph type"),
                            dcc.Dropdown(id="graph_type", value="scatter",
                                         options=[
                                                  {"label": "Bar Chart", "value": "bar"},
                                                  {"label": "Area Plot Chart", "value": "line"},
                                                  {"label": "Line Chart", "value": "area"},
                                                  {"label": "Scatter Chart", "value": "scatter"},
                                                  {"label": "Histogram", "value": "histogram_1"}
                                                  ])
                        ])
                    ])
                ]),
                dbc.Row([
                    dbc.Col([
                        dbc.FormGroup([
                            html.Label("Sample Records"),
                            dcc.Slider(
                                id="data_display",
                                min=0,
                                max=row_count,
                                value=1000,
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ])
                    ])
                ]),
                dbc.Row([
                    dbc.Col([
                        dcc.Graph(id='graph')
                    ])
                ])
            ])
        ]),
        dcc.Tab(label="Trends", children=[
                        dbc.Container([
                            html.Br(),
                            html.H1("Important Trends in the Dataset", style={"margin-bottom": '1em', "text-align": "center"}),
                            html.P("Select an animation:"),
                            dcc.RadioItems(
                                id='selection',
                                options=[{'label': x, 'value': x} for x in animations],
                                value='Scatter'
                            ),
                            dcc.Graph(id="graph1", style={"height": "80vh", "width": "100%"}),
                        ])
                    ]),
        dcc.Tab(label="Prediction", children=[
                        dbc.Container([
                            html.Br(),
                            html.H2(children='Select Manufacturer, Model and Year ', style={'text-align': 'center'}),
                            html.Hr(),
                            dbc.CardDeck([card_manufacturer, card_model, card_year]),
                            html.Br(),
                            html.H2(children='Apply more Filters '),
                            dbc.CardDeck([card_fuel_type, card_odometer_reading, card_vehicle_status, card_transmission_type]),
                            html.Br(),
                            dbc.CardDeck([card_drivetrain, card_vehicle_type, card_paint_color]),
                            html.Hr(),
                            dbc.CardDeck([card_model_selection, card_predicted_price]),
                            html.Br(),
                            dbc.CardDeck([card_lower_range, card_upper_range]),
                            html.Br(),
                            dbc.CardDeck([card_feature_importance]),
                            html.Br(),
                        ])
                    ])
    ])
])


def replace_categorical_keys(input, dictionary):
    for key, value in dictionary.items():
        if key == input:
            input = input.replace(key, str(value))
    return input


def replace_numeric_keys(input, dictionary):
    for key, value in dictionary.items():
        if str(key) == input:
            input = input.replace(str(key), str(value))
    return input


def year_dictionary(input):
    year_dict = dict(zip(before_label_encoding.year, after_label_encoding.year))
    modified_input = replace_numeric_keys(input, year_dict)
    return modified_input


def manufacturer_dictionary(input):
    manufacturer_dict = dict(zip(before_label_encoding.manufacturer, after_label_encoding.manufacturer))
    modified_input = replace_categorical_keys(input, manufacturer_dict)
    return modified_input


def model_dictionary(input):
    model_dict = dict(zip(before_label_encoding.model, after_label_encoding.model))
    modified_input = replace_categorical_keys(input, model_dict)
    return modified_input


def fuel_dictionary(input):
    fuel_dict = dict(zip(before_label_encoding.fuel, after_label_encoding.fuel))
    modified_input = replace_categorical_keys(input, fuel_dict)
    return modified_input


def odometer_dictionary(input):
    odometer_dict = dict(zip(before_label_encoding.odometer, after_label_encoding.odometer))
    modified_input = replace_numeric_keys(input, odometer_dict)
    return modified_input


def title_status_dictionary(input):
    title_status_dict = dict(zip(before_label_encoding.title_status, after_label_encoding.title_status))
    modified_input = replace_categorical_keys(input, title_status_dict)
    return modified_input


def transmission_dictionary(input):
    transmission_dict = dict(zip(before_label_encoding.transmission, after_label_encoding.transmission))
    modified_input = replace_categorical_keys(input, transmission_dict)
    return modified_input


def train_type_dictionary(input):
    train_type_dict = dict(zip(before_label_encoding.drive, after_label_encoding.drive))
    modified_input = replace_categorical_keys(input, train_type_dict)
    return modified_input


def vehicle_type_dictionary(input):
    vehicle_type_dict = dict(zip(before_label_encoding.type, after_label_encoding.type))
    modified_input = replace_categorical_keys(input, vehicle_type_dict)
    return modified_input


def paint_dictionary(input):
    paint_dict = dict(zip(before_label_encoding.paint_color, after_label_encoding.paint_color))
    modified_input = replace_categorical_keys(input, paint_dict)
    return modified_input


def predict_price(ml_model, manufacturer_pp, model_pp, purchase_year_pp, fuel_type_pp, odometer_pp, vehicle_status_pp,
                  transmission_type_pp, train_type_pp, vehicle_type_pp, paint_pp):
    prediction_df = pd.DataFrame({'year_p': [purchase_year_pp], 'manufacturer_p': [manufacturer_pp],
                                  'model_p': [model_pp], 'fuel_p': [fuel_type_pp], 'odometer_p': [odometer_pp],
                                  'title_status_p': [vehicle_status_pp], 'transmission_p': [transmission_type_pp],
                                  'drive_p': [train_type_pp], 'type_p': [vehicle_type_pp], 'paint_color_p': [paint_pp]})

    prediction_df['year_p'] = prediction_df['year_p'].astype(int)
    prediction_df['year_p'] = year_dictionary(purchase_year_pp)
    prediction_df['manufacturer_p'] = manufacturer_dictionary(manufacturer_pp)
    prediction_df['model_p'] = model_dictionary(model_pp)
    prediction_df['fuel_p'] = fuel_dictionary(fuel_type_pp)
    prediction_df['odometer_p'] = prediction_df['odometer_p'].astype(int)
    prediction_df['odometer_p'] = odometer_dictionary(odometer_pp)
    prediction_df['title_status_p'] = title_status_dictionary(vehicle_status_pp)
    prediction_df['transmission_p'] = transmission_dictionary(transmission_type_pp)
    prediction_df['drive_p'] = train_type_dictionary(train_type_pp)
    prediction_df['type_p'] = vehicle_type_dictionary(vehicle_type_pp)
    prediction_df['paint_color_p'] = paint_dictionary(paint_pp)

    data = prediction_df.to_numpy()

    rf_Model = pickle.load(open('Random_Forest_Model1.pkl', 'rb'))
    price = rf_Model.predict(data)

    return price


# callback to display data in table based on different inputs
@app.callback(
    [Output('dataframe_table', 'data'),
     Output('dataframe_table', 'columns'),
     Output('dataframe_table', 'page_size')],
    [Input('display_data', 'n_clicks')],
    [State('records_on_single_page', 'value'),
     State('fields_dropdown', 'value')]
)
def update_data_table(n_clicks, rows, columns):
    df = vehicle_df[columns]
    selected_columns = [{'id': i, 'name': str.upper(i)} for i in columns]
    records_per_page = rows

    return [df.to_dict('records'), selected_columns, records_per_page]

# callback to return the column values
@app.callback(
    Output('select_value', 'options'),
    Input('filter_column', 'value'),
    prevent_initial_call=True
)
def get_column_values(column_name):
    if column_name != 'None':
        return [{"label": i, "value": i} for i in np.unique(vehicle_df[column_name])]
    else:
        return [{"label": "None", "value": "None"}]

# callback for slider max and value
@app.callback(
    [Output('data_display', 'max'),
     Output('data_display', 'value')],
    [Input('filter_column', 'value'),
     Input('select_value', 'value')]
)
def update_slider(filter_column, filter_value):
    if filter_column != 'None':
        if filter_value != 'None':
            df_new = vehicle_df[vehicle_df[filter_column] == filter_value]
            count = np.count_nonzero(df_new.index)
            return [count, count]
        else:
            return [row_count, 1000]
    else:
        return [row_count, 1000]

# callback to show different graphs based on input
@app.callback(
    Output('visual_graph', 'figure'),
    Input('visualization_button', 'n_clicks'),
    State('visualization_type', 'value')
)
def update_visualization(n_clicks, visual_type):
    if visual_type == 'avg_pr':
        return average_price_based_on_fuel_type()
    elif visual_type == 'correlation_heatmap':
        return correlation_heatmap()
    elif visual_type == 'pie_chart':
        return pie_chart_showing_number_of_models_per_manufacturer()
    elif visual_type == 'histogram':
        return histogram_showing_cars_at_different_price()
    elif visual_type == 'cars_available_per_year':
        return chart_showing_number_of_cars_available_per_year()
    elif visual_type == 'available_color':
        return available_cars_based_on_color()
    elif visual_type == 'available_type':
        return available_models_based_on_type()

# callback to show graph based on user selection
@app.callback(
    Output('graph', 'figure'),
    [Input('x_axis', 'value'),
     Input('y_axis', 'value'),
     Input('filter_column', 'value'),
     Input('select_value', 'value'),
     Input('graph_type', 'value'),
     Input('data_display', 'value')]
)
def update_graph(x_col, y_col, filter_column, filter_value, graph_type, num_of_records):
    df = vehicle_df.copy(deep=True)

    if filter_column != 'None':
        if filter_value != 'None':
            df = df[df[filter_column] == filter_value]

    df = df.head(num_of_records)

    if graph_type == 'bar':
        return px.bar(df, x=x_col, y=y_col)
    elif graph_type == 'scatter':
        return px.scatter(df, x=x_col, y=y_col)
    elif graph_type == 'line':
        df = df.sort_values(by=[x_col])
        return px.line(df, x=x_col, y=y_col)
    elif graph_type == 'area':
        return px.area(df, x=x_col, y=y_col)
    elif graph_type == 'histogram_1':
        return px.histogram(df, x=x_col, y=y_col)

# CALLBACK for Animations
@app.callback(
    Output("graph1", "figure"),
    [Input("selection", "value")])
def display_animated_graph(s):
    return animations[s]


# CALLBACK to predict price based on user inputs
@app.callback(
    Output(component_id='predicted_price', component_property='children'),
    Output(component_id='lower_range', component_property='children'),
    Output(component_id='upper_range', component_property='children'),
    Input(component_id='predict_price', component_property='n_clicks'),
    [State('ml_models_dropdown', 'value'),
     State('manufacturer_dropdown', 'value'),
     State('model_dropdown', 'value'),
     State('year_dropdown', 'value'),
     State('fuel_type_dropdown', 'value'),
     State('odometer_reading', 'value'),
     State('vehicle_status_dropdown', 'value'),
     State('transmission_dropdown', 'value'),
     State('traintype_dropdown', 'value'),
     State('vehicle_type_dropdown', 'value'),
     State('paint_dropdown', 'value')],
    prevent_initial_call=True
)
def user_preference(n_clicks, ml_model, manufacturer_p, model_p, purchase_year_p, fuel_type_p, odometer_p,
                    vehicle_status_p, transmission_type_p, train_type_p, vehicle_type_p, paint_p):

    predicted_price = predict_price(ml_model, manufacturer_p, model_p, purchase_year_p, fuel_type_p, int(odometer_p),
                                    vehicle_status_p, transmission_type_p, train_type_p, vehicle_type_p, paint_p)

    upper_range = (predicted_price + (.10 * predicted_price))
    lower_range = (predicted_price - (.10 * predicted_price))
    upper_range = int(upper_range)
    lower_range = int(lower_range)

    return numpy.round(predicted_price, 2), lower_range, upper_range

output_col = "price"
feature_cols = after_label_encoding.columns.tolist()
feature_cols.remove(output_col)
x_features = after_label_encoding[feature_cols]
y_label = after_label_encoding[output_col]

# Callback to check Feature Importance
@app.callback(
    Output(component_id='feature_importance', component_property='figure'),
    [Input(component_id='ml_models_dropdown', component_property='value')]
)
def feature_importance(selected_model):
    rf_Model = pickle.load(open('Random_Forest_Model1.pkl', 'rb'))
    importances = rf_Model.feature_importances_
    features = x_features.columns
    x_values = list(features)
    fig = px.bar(x=x_values, y=importances, title='Random Forest Variables Importance',
                 labels={'x': 'Features', 'y': 'Feature Weightage'})
    fig.update_xaxes(tickangle=90, tickmode='array', tickvals=features)

    return fig


# <------------ CALLBACK --------------->
# This Callback is used to change values inside Model column based on Manufacturer selected.
@app.callback(
    Output(component_id='model_dropdown', component_property='options'),
    Input(component_id='manufacturer_dropdown', component_property='value')
)
def choose_model(man):
    df = vehicle_df.copy(deep=True)
    df = df[df['manufacturer'] == man]
    return [{'label': i, 'value': i} for i in np.unique(df['model'].values)]


if __name__ == "__main__":
    app.run_server(host='0.0.0.0',port=8080)
