import pandas as pd
from climate_database import query_climate_database
from climate_database import query_regional_patterns
from climate_database import query_regional_warming
from plotly import express as px
from sklearn.linear_model import LinearRegression
import calendar

def temperature_coefficient_plot(db_file, country, year_begin, year_end, month, min_obs, **kwargs):
    """
    Parameters:
    -----------
    db_file (str): File name for the database
    country (str): Name of the country
    year_begin (int): Earliest year for the date range (inclusive)
    year_end (int): Latest year for the date range (inclusive)
    month (int): Month of the year
    min_obs (int): Minimum number of years of data required for a station to be included
    **kwargs: Additional keyword arguments to pass to px.scatter_mapbox()
    
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    
    # Query the database
    df = query_climate_database(db_file, country, year_begin, year_end, month)
    
    # Filter for stations with minimum required observations
    df = df.groupby(['NAME', 'LATITUDE', 'LONGITUDE']).filter(lambda x: len(x) >= min_obs)
    
    # Calculate temperature coefficients using linear regression
    def calc_coefficient(group):
        X = group['Year'].values.reshape(-1, 1)
        y = group['Temp'].values
        model = LinearRegression()
        model.fit(X, y)
        return model.coef_[0]
    
    coefficients = df.groupby(['NAME', 'LATITUDE', 'LONGITUDE'])\
        .apply(calc_coefficient)\
        .reset_index(name='temp_change')
    
    # Get month name
    month_name = calendar.month_name[month]
    
    # Create the interactive map
    fig = px.scatter_mapbox(
        coefficients,
        lat='LATITUDE',
        lon='LONGITUDE',
        color='temp_change',
        hover_name='NAME',
        hover_data={
            'LATITUDE': ':.3f',
            'LONGITUDE': ':.2f',
            'temp_change': ':.4f'
        },
        labels={
            'temp_change': 'Estimated Yearly Increase (°C)',
            'LATITUDE': 'LATITUDE',
            'LONGITUDE': 'LONGITUDE'
        },
       title=f'Estimates of yearly increase in temperature in {month_name}<br>for stations in {country}, years {year_begin} - {year_end}',
		range_color=[-0.1, 0.1],
		**kwargs
    )
    
    return fig


def plot_regional_patterns(db_file, country, year_begin, year_end):
    """
    Parameters:
    -----------
    db_file (str): File name for the database
    country (str): Name of the country
    year_begin (int): Earliest year for the date range (inclusive)
    year_end (int): Latest year for the date range (inclusive)
    
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    # Query the data
    df = query_regional_patterns(db_file, country, year_begin, year_end)
    
    # Add month names
    df['MonthName'] = df['Month'].apply(lambda x: calendar.month_abbr[x])
    
    # Calculate regional averages by month
    monthly_avgs = df.groupby(['region', 'Month', 'MonthName'])['Temp'].mean().reset_index()
    
    # Create faceted line plot
    fig = px.line(
        monthly_avgs,
        x='MonthName',
        y='Temp',
        facet_col='region',
        title=f'Monthly Temperature Patterns by Region in {country} ({year_begin}-{year_end})',
        labels={
            'MonthName': 'Month',
            'Temp': 'Average Temperature (°C)',
            'region': 'Region'
        },
        height=500
    )
    
    return fig


def plot_regional_warming(db_file, country, year_begin, year_end):
    """
    Parameters:
    -----------
    db_file (str): File name for the database
    country (str): Name of the country
    year_begin (int): Earliest year for the date range (inclusive)
    year_end (int): Latest year for the date range (inclusive)
    
    Returns:
    --------
    plotly.graph_objects.Figure
    """
    # Query data
    df = query_regional_warming(db_file, country, year_begin, year_end)
    
    # Create faceted scatter plot with trend lines
    fig = px.scatter(
        df,
        x='Year',
        y='Temp',
        color='season',
        facet_col='region',
        trendline="ols",
        title=f'Summer vs Winter Warming Rates by Region in {country} ({year_begin}-{year_end})',
        labels={
            'Year': 'Year',
            'Temp': 'Temperature (°C)',
            'season': 'Season',
            'region': 'Region'
        },
        height=500
    )
    
    return fig