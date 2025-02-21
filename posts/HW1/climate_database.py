import pandas as pd
import sqlite3

def query_climate_database(db_file, country, year_begin, year_end, month):
    """
    Parameters:
    -----------
    db_file (str): File name for the database
    country (str): Name of the country
    year_begin (int): Earliest year for the date range (inclusive)
    year_end (int): Lastest year for the date range (inclusive)
    month (int): Month of the year
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing temperature readings with columns:
        NAME, LATITUDE, LONGITUDE, Country, Year, Month, Temp
    """
    
    # Create connection to database
    conn = sqlite3.connect(db_file)
    
    # Construct the SQL query using f-strings
    query = f"""
        SELECT 
            s.NAME,
            s.LATITUDE,
            s.LONGITUDE,
            c.Name as Country,
            t.Year,
            t.Month,
            t.Temp
        FROM temperatures t
        JOIN stations s ON t.ID = s.ID
        JOIN countries c ON substr(t.ID, 1, 2) = c."FIPS 10-4"
        WHERE c.Name = '{country}'
        AND t.Year >= {year_begin}
        AND t.Year <= {year_end}
        AND t.Month = {month}
        ORDER BY s.NAME, t.Year, t.Month
    """
    
    # Execute query and load results into DataFrame
    df = pd.read_sql_query(query, conn)
    
    # Close database connection
    conn.close()
    
    return df


def query_regional_patterns(db_file, country, year_begin, year_end):
    """
    Parameters:
    -----------
    db_file (str): File name for the database
    country (str): Name of the country
    year_begin (int): Earliest year for the date range (inclusive)
    year_end (int): Latest year for the date range (inclusive)
    
    Returns:
    --------
    pandas.DataFrame with columns:
        Year: year of measurement
        Month: month of measurement
        region: geographical region (North, Central, South)
        temperature: average temperature for that region/month/year
    """
    conn = sqlite3.connect(db_file)
    
    query = f"""
        WITH station_regions AS (
            SELECT 
                s.ID,
                CASE 
                    WHEN s.LATITUDE >= LAT_MAX - (LAT_MAX - LAT_MIN)/3 THEN 'North'  -- North region if latitude is at the upper third of the range
                    WHEN s.LATITUDE >= LAT_MAX - 2*(LAT_MAX - LAT_MIN)/3 THEN 'Central'  -- Central region if latitude is in the middle third
                    ELSE 'South'   -- South region for the remaining latitudes
                END as region
            FROM stations s
            JOIN countries c ON substr(s.ID, 1, 2) = c."FIPS 10-4"
            CROSS JOIN (
                SELECT 
                    MAX(LATITUDE) as LAT_MAX,
                    MIN(LATITUDE) as LAT_MIN
                FROM stations s2
                JOIN countries c2 ON substr(s2.ID, 1, 2) = c2."FIPS 10-4"
                WHERE c2.Name = '{country}'
            )
            WHERE c.Name = '{country}'
        )
        SELECT 
            t.Year,
            t.Month,
            sr.region,
            t.Temp
        FROM temperatures t
        JOIN station_regions sr ON t.ID = sr.ID
        JOIN countries c ON substr(t.ID, 1, 2) = c."FIPS 10-4"
        WHERE c.Name = '{country}'
        AND t.Year >= {year_begin}
        AND t.Year <= {year_end}
        GROUP BY t.Year, t.Month, sr.region
        ORDER BY region, t.Year, t.Month
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def query_regional_warming(db_file, country, year_begin, year_end):
    """
    Parameters:
    -----------
    db_file (str): File name for the database
    country (str): Name of the country
    year_begin (int): Earliest year for the date range (inclusive)
    year_end (int): Latest year for the date range (inclusive)
    
    Returns:
    --------
    pandas.DataFrame with regional summer/winter temperatures over time
    """
    conn = sqlite3.connect(db_file)
    
    query = f"""
        WITH station_regions AS (
            SELECT 
                s.ID,
                CASE 
                    WHEN s.LATITUDE >= LAT_MAX - (LAT_MAX - LAT_MIN)/3 THEN 'North'
                    WHEN s.LATITUDE >= LAT_MAX - 2*(LAT_MAX - LAT_MIN)/3 THEN 'Central'
                    ELSE 'South'
                END as region
            FROM stations s
            JOIN countries c ON substr(s.ID, 1, 2) = c."FIPS 10-4"
            CROSS JOIN (
                SELECT 
                    MAX(LATITUDE) as LAT_MAX,
                    MIN(LATITUDE) as LAT_MIN
                FROM stations s2
                JOIN countries c2 ON substr(s2.ID, 1, 2) = c2."FIPS 10-4"
                WHERE c2.Name = '{country}'
            )
            WHERE c.Name = '{country}'
        )
        SELECT 
            t.Year,
            t.Month,
            CASE 
                WHEN t.Month IN (6,7,8) THEN 'Summer'
                WHEN t.Month IN (12,1,2) THEN 'Winter'
            END as season,
            sr.region,
            t.Temp
        FROM temperatures t
        JOIN station_regions sr ON t.ID = sr.ID
        JOIN countries c ON substr(t.ID, 1, 2) = c."FIPS 10-4"
        WHERE c.Name = '{country}'
        AND t.Year >= {year_begin}
        AND t.Year <= {year_end}
        AND t.Month IN (1,2,6,7,8,12)  -- Summer and winter months only
        GROUP BY t.Year, t.Month, sr.region
        ORDER BY t.Year, t.Month, sr.region
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df