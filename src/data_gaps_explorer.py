# tools to explore data gaps
import pandas as pd
################################################
# 1. explore temporal data gaps in Eurostat data
################################################
def first_year(df, city, attribute):
    """get first year for which data is available. First possible year is 1991

    Args:
        df (pandas DataFrame): dataframe pulled from eurostat cube. E.g., with df = pd.read_sql_query("SELECT * FROM c_urban_cube_eurostat", con)
        city (string): Eurostat city code, that is, `urau_code` attribute in the Eurostat cube
        attribute (string): Eurostat indicator code, that is, `indic_code` attribute in the Eurostat cube

    Returns:
        int: first year for which data is available for given city and indicator. None if no data is available
    """
    line = df[(df['urau_code'] == city) & (df['indic_code'] == attribute)]
    if(len(line)==0):
        return None
    first_year = 1991
    col_set = line.columns
    col_set = col_set.drop(['index', 'indic_code', 'urau_code'])
    for col in col_set:
        if  not pd.isnull(line.iloc[0][col]): 
            first_year = col
            break
    return first_year

def last_year(df, city, attribute):
    """get last year for which data is available. Latest possible year is 2021.

    Args:
        df (pandas DataFrame): dataframe pulled from eurostat cube. E.g., with df = pd.read_sql_query("SELECT * FROM c_urban_cube_eurostat", con)
        city (string): Eurostat city code, that is, `urau_code` attribute in the Eurostat cube
        attribute (string): Eurostat indicator code, that is, `indic_code` attribute in the Eurostat cube

    Returns:
        int: last year for which data is available for given city and indicator. None if no data is available
    """
    line = df[(df['urau_code'] == city) & (df['indic_code'] == attribute)]
    if(len(line)==0):
        return None
    last_year = 2021
    col_set = line.columns[::-1]
    col_set = col_set.drop(['index', 'indic_code', 'urau_code'])
    for col in col_set:
        if  not pd.isnull(line.iloc[0][col]): 
            last_year = col
            break
    return last_year

def isAvailable(df, city, attribute, sequence):
    """Check if data is available every `sequence` years

    Args:
        df (pandas DataFrame): dataframe pulled from eurostat cube. E.g., with df = pd.read_sql_query("SELECT * FROM c_urban_cube_eurostat", con)
        city (string): Eurostat city code, that is, `urau_code` attribute in the Eurostat cube
        attribute (string): Eurostat indicator code, that is, `indic_code` attribute in the Eurostat cube
        sequence (int): step size
    Returns:
        boolean: True if data is available every `sequence` years, else False
    """
    line = df[(df['urau_code'] == city) & (df['indic_code'] == attribute)]
    if(len(line)==0):
        return False
    year = first_year(df, city, attribute)
    last = last_year(df, city, attribute)
    col_set = df.columns
    for col in col_set:
        if col == year:
            break
        col_set = col_set.drop(col)
    
    for col in col_set[::-1]:
        if col == last:
            break
        col_set = col_set.drop(col)
    last = last_year(df, city, attribute)
    available = True
    itterate = col_set
    for i in range(sequence):
        itterate = itterate[:-1]
    for idx, col in enumerate(itterate):
        if pd.isnull(line.iloc[0][col_set[idx+sequence]]):
            available = False
            break
    return available

def number_years(df, city, attribute):
    """Count number of years with data, given city and indicator

    Args:
        df (pandas DataFrame): dataframe pulled from eurostat cube. E.g., with df = pd.read_sql_query("SELECT * FROM c_urban_cube_eurostat", con)
        city (string): Eurostat city code, that is, `urau_code` attribute in the Eurostat cube
        attribute (string): Eurostat indicator code, that is, `indic_code` attribute in the Eurostat cube

    Returns:
        int: number of years with non-null data for given city and indicator
    """
    line = df[(df['urau_code'] == city) & (df['indic_code'] == attribute)]
    if(len(line)==0):
        return 0
    return len(line.columns) - line.isna().sum().sum() - 3

def get_data_gaps(ts_table):
    """Retrieve information about data gaps in a time series. The time series is structured as a pandas DataFrame, where each column is a time step (e.g. years). 
    No other columns are allowed. The function returns a dataframe with the same indices of the original dataset and the columns 
    - `first_year`: first time step where data is available
    - `last_year`: last time step where data is available,
    - `n_years`: number of time steps with non-null data,
    - `gap_max`: maximum length of consecutive null data,
    - `gap_median`: median length of consecutive null data
    Args:
        ts_table (pandas.DataFrame): input time series

    Returns:
        pandas.DataFrame: DataFrame of the same length of the original one, with columns `first_year`, `last_year`, `n_years`, `gap_max`, `gap_median`
    """
    av_matrix = ~pd.isnull(ts_table)
    av_matrix = av_matrix.astype(int)
    yr_with_values = av_matrix.apply(lambda x: np.where(x.values == 1)[0], axis=1)
    n_years = yr_with_values.apply(lambda x: len(x))
    first_year = yr_with_values.apply(lambda x: 1991+x[0] if len(x) > 0 else 0)
    last_year = yr_with_values.apply(lambda x: 1991+x[-1] if len(x) > 0 else 0)
    gaps = yr_with_values.apply(lambda x: [t - s for s, t in zip(x, x[1:])])
    gaps_max = gaps.apply(lambda x: np.max(x) if len(x) > 0 else 0)
    gaps_median = gaps.apply(lambda x: int(math.ceil(np.median(x))) if len(x) > 0 else 0)
    d = {
        "first_year": first_year,
        "last_year": last_year,
        "n_years": n_years,
        "gap_max": gaps_max,
        "gap_median": gaps_median
    }
    df = pd.DataFrame(data=d)
    return df