from helper_methods import *


def process_file():

    # reading data from vehicles dataset
    vehicles_df = read_csv_file(Path('Data', 'vehicles.csv'))

    # removing the rows where vehicle price is less than 100
    vehicles_df = vehicles_df[vehicles_df['price'] > 100]

    # keeping only the required columns
    vehicles_df = vehicles_df[['price', 'year', 'manufacturer', 'model', 'fuel', 'odometer', 'title_status',
                               'transmission', 'drive', 'type', 'paint_color']]

    # getting the numeric columns
    num_columns = get_numeric_columns(vehicles_df)

    # fixing the nan
    for col in vehicles_df.columns:
        if col == 'year':
            vehicles_df.dropna(subset=[col], inplace=True)
        else:
            vehicles_df = fix_nans(vehicles_df, col)

    # converting the year column data type to integer
    vehicles_df['year'] = vehicles_df['year'].astype('int32')

    # removing rows with invalid characters
    vehicles_df = remove_invalid_rows(vehicles_df, 'model')

    # fixing the outliers
    for column in num_columns:
        vehicles_df = fix_outlier(vehicles_df, column)

    write_csv_file(vehicles_df, Path('Data', 'vehicle_final.csv'))

    return vehicles_df


def process_file_again():

    # reading the file processed in process_file
    vehicles_df = process_file()

    # getting the list of numeric columns
    num_columns = get_numeric_columns(vehicles_df)

    # normalizing the columns
    for col in num_columns:
        if col != 'price':
            vehicles_df.loc[:, col] = normalize_column(vehicles_df.loc[:, col])

    # getting the list of categorical columns
    categorical_column = get_text_categorical_columns(vehicles_df)

    # data encoding
    for column in categorical_column:
        le = generate_label_encoder(vehicles_df.loc[:, column])
        vehicles_df = replace_with_label_encoder(vehicles_df, column, le)

    write_csv_file(vehicles_df, Path('Data', 'vehicle_final_le.csv'))

    return vehicles_df

