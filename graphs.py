import plotly.express as px
from file_processing import *


# method to read file
def read_data_file():
    df = read_csv_file(Path('Data', 'vehicle_final.csv'))
    return df


# method to show bar graph providing average price based on fuel type
def average_price_based_on_fuel_type():
    df = read_data_file()
    df1 = df.copy(deep=True)
    df1 = df1.groupby('fuel').agg({'price': 'mean'}).reset_index()
    fig = px.bar(data_frame=df1, x='fuel', y='price', color='fuel')
    fig.update_layout(xaxis_title="fuel", yaxis_title="average price")
    return fig


# method to show total number of available vehicles based on type
def available_models_based_on_type():
    df = read_data_file()
    df1 = df.groupby('type').agg({'model': 'count'}).reset_index()
    df1 = df1.sort_values(by='model')
    fig = px.bar(data_frame=df1, x='type', y='model', color='type')
    fig.update_layout(xaxis_title="type", yaxis_title="number of available vehicles")
    return fig


# method to show number of cars available based on color
def available_cars_based_on_color():
    df = read_data_file()
    df1 = df.groupby('paint_color').agg({'model': 'count'}).reset_index()
    df1 = df1.sort_values(by='model', ascending=False)
    fig = px.bar(data_frame=df1, x='paint_color', y='model')
    fig.update_layout(xaxis_title="paint color", yaxis_title="number of available cars")
    return fig


# method to get heatmap of correlation between columns of dataset
def correlation_heatmap():
    df1 = read_csv_file(Path('Data', 'vehicle_final_le.csv'))
    fig = px.imshow(df1.corr(), color_continuous_scale='OrRd')
    return fig


# method to draw pie chart showing number of models per manufacturer
def pie_chart_showing_number_of_models_per_manufacturer():
    df = read_data_file()
    df1 = df[['manufacturer', 'model']]
    df1 = df1.drop_duplicates()
    df2 = df1.groupby('manufacturer').agg({'model': 'count'}).reset_index()
    fig = px.pie(df2, names='manufacturer', values='model')
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig


# method to get histogram showing number of cars available at different price ranges
def histogram_showing_cars_at_different_price():
    df = read_data_file()
    df1 = df.groupby('price').agg({'model': 'count'}).reset_index()
    fig = px.histogram(df1, x='price', y='model')
    fig.update_layout(xaxis_title="price", yaxis_title="number of cars")
    return fig


# method to get bar chart showing number of cars available per year
def chart_showing_number_of_cars_available_per_year():
    df = read_data_file()
    df1 = df.groupby('year').agg({'model': 'count'}).reset_index()
    df1 = df1.sort_values(by=['year'])
    fig = px.bar(df1, x='year', y='model', color='year', color_continuous_scale='YlOrBr')
    fig.update_layout(xaxis_title="year", yaxis_title="number of cars")
    return fig

