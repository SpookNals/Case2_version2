import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression

# conda install conda-forge::folium
import folium
# conda install conda-forge::streamlit-folium
from streamlit_folium import folium_static

st.set_page_config(
    page_title="Weer in Amsterdam",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load data
df = pd.read_csv('weather_cleaned.csv')
df_raw = pd.read_csv('weather.csv')
df['dt'] = pd.to_datetime(df['dt'])

# create a screen switcher
st.sidebar.radio(
        "Selecteer wat je wil zien",
        key="visibility",
        options=["api", "visuals"],
    )
st.sidebar.write('---')

if st.session_state.visibility == "visuals":

    # Create a dropdown menu for selecting a year
    available_years = list(range(df['year'].min(), df['year'].max() + 1))

    # Create a slider for selecting a year, with an "All years" option
    year = st.sidebar.slider('Selecteer een Jaar', min_value=min(available_years), max_value=max(available_years), value=min(available_years), step=1, format="%d")

    # Create a checkbox that selects all years
    all_years = st.sidebar.checkbox('Laat alle Jaren zien')

    # create a dropdown menu for selecting celsius or fahrenheit
    unit = st.sidebar.selectbox('Selecteer een unit', ['Celsius', 'Kelvin', 'Fahrenheit'])

    # add a line of text to the sidebar
    st.sidebar.write('Selecteer meer plot opties:')

    # Create a checkbox to show the 'feels_like' temperature
    show_feels_like = st.sidebar.checkbox('Feels Like')

    # Create a checkbox to show the 'dew_point' temperature
    show_dew_point = st.sidebar.checkbox('Dew Point')

    # Create a checkbox for showing the trendline
    show_trendline = st.sidebar.checkbox('Trendline')
    st.sidebar.write('---')

    # Filter the data based on the selected year
    if all_years:
        df_filtered = df
    else:
        df_filtered = df[df['dt'].dt.year == year]

    # Convert temperature to the selected unit
    if unit == 'Fahrenheit':
        temp_unit = 'fahrenheit'
        df_filtered['feels_like'] = (df_filtered['feels_like'] - 273.15) * 9/5 + 32
        df_filtered['dew_point'] = (df_filtered['dew_point'] - 273.15) * 9/5 + 32
    elif unit == 'Kelvin':
        temp_unit = 'kelvin'
        pass
    else:
        temp_unit = 'celsius'
        df_filtered['feels_like'] = df_filtered['feels_like'] - 273.15
        df_filtered['dew_point'] = df_filtered['dew_point'] - 273.15

    # make the y-axes range static for each unit
    if unit == 'Fahrenheit':
        y_range = [0, 100]
    elif unit == 'Kelvin':
        y_range = [250, 320]
    else:
        y_range = [-20, 40]

    # Create a line chart for the selected year
    title_fig_1 = f'Temperatuur Amsterdam in {year}' if not all_years else 'Temperatuur Amsterdam'
    fig = px.line(df_filtered, x="dt", y=temp_unit, title=title_fig_1)
    fig.update_traces(name='Temperatuur', showlegend=True)
    fig.update_layout(
        xaxis_title='Tijd', 
        yaxis_title=unit,
        width=900,
        height=500,
        yaxis=dict(range=y_range),  # Set the y-axis range inside the yaxis dictionary
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    # Get the index of the maximum and minimum temperatures for each year
    max_temp_idx = df_filtered.groupby('year')[temp_unit].idxmax()
    min_temp_idx = df_filtered.groupby('year')[temp_unit].idxmin()

    # Use these indices to get the corresponding dates and temperatures
    max_temp_per_year = df_filtered.loc[max_temp_idx]
    min_temp_per_year = df_filtered.loc[min_temp_idx]

    # Add the max temperature per year to the chart (on the correct dates)
    fig.add_scatter(
        x=max_temp_per_year['dt'], 
        y=max_temp_per_year[temp_unit], 
        mode='markers', 
        name='Max Temperatuur', 
        marker=dict(color='red'),
        marker_size=5
    )

    # Add the min temperature per year to the chart (on the correct dates)
    fig.add_scatter(
        x=min_temp_per_year['dt'], 
        y=min_temp_per_year[temp_unit], 
        mode='markers', 
        name='Min Temperatuur', 
        marker=dict(color='blue'),
        marker_size=5
    )
    # Add the 'trendline' to the chart if the user selected it
    if show_trendline:
        X = np.arange(len(df_filtered)).reshape(-1, 1)  # Indices for trendline
        Y = df_filtered[temp_unit].values  # Temperatures
        reg = LinearRegression().fit(X, Y)
        df_filtered['bestfit'] = reg.predict(X)
        fig.add_scatter(x=df_filtered['dt'], y=df_filtered['bestfit'], mode='lines', name='Trendline', line=dict(color='yellow'))

    # Add the 'feels_like' temperature to the chart if the user selected it
    if show_feels_like:
        fig.add_scatter(x=df_filtered['dt'], y=df_filtered['feels_like'], opacity=0.5, mode='lines', name='Feels Like',
                        line=dict(color='red'))

    # Add the 'dew_point' temperature to the chart if the user selected it
    if show_dew_point:
        fig.add_scatter(x=df_filtered['dt'], y=df_filtered['dew_point'], opacity=0.5, mode='lines', name='Dew Point',
                        line=dict(color='green'))

    # Create a new 'year-month' column to represent the months in the format 'YYYY-MM'
    df_filtered['year_month'] = df_filtered['dt'].dt.to_period('M').astype(str)

    # Group the data by the new 'year_month' column to calculate the average temperature per month
    df_grouped = df_filtered.groupby('year_month').mean(numeric_only=True).reset_index()

    # set dynamic variables for the title and color of the bar chart
    title_fig_2 = f'Gemiddelde temperatuur per maand in {year}' if not all_years else 'Gemiddelde Temperatuur per Maand'
    fig2_color = 'month' if not all_years else 'year'

    fig_2 = px.bar(df_grouped, x='year_month',color=fig2_color,  y=temp_unit, title=title_fig_2)

    # Update traces
    fig_2.update_traces(name='Temperatuur', showlegend=True)
    fig_2.update_layout(
        xaxis_title='Tijd',
        yaxis_title=f'{unit}',
        width=900,
        height=500,
        yaxis=dict(range=y_range),
        xaxis=dict(type='category'),
        barmode='overlay',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    if show_trendline:
        X = np.arange(len(df_grouped)).reshape(-1, 1)  # Indices for trendline
        Y = df_grouped[temp_unit].values  # Average temperatures
        reg = LinearRegression().fit(X, Y)
        df_grouped['bestfit'] = reg.predict(X)
        fig_2.add_scatter(x=df_grouped['year_month'], y=df_grouped['bestfit'], mode='lines', name='Trendline', line=dict(color='yellow'))

    if show_feels_like:
        df_grouped_feels_like = df_filtered.groupby('year_month').mean(numeric_only=True).reset_index()
        fig_2.add_bar(x=df_grouped_feels_like['year_month'], y=df_grouped_feels_like['feels_like'], opacity=0.5, name='Feels Like',
                    marker_color='red')

    if show_dew_point:
        df_grouped_dew_point = df_filtered.groupby('year_month').mean(numeric_only=True).reset_index()
        fig_2.add_bar(x=df_grouped_dew_point['year_month'], y=df_grouped_dew_point['dew_point'], opacity=0.5, name='Dew Point',
                    marker_color='green') 
    
    # create a fig3 that shows the average temperature of each year in a bar chart
    df_grouped_year = df_filtered.groupby('year').mean(numeric_only=True).reset_index()
    fig3_color = None if not all_years else 'year'
    fig3_title = f'Gemiddelde temperatuur in {year}' if not all_years else 'Gemiddelde Temperatuur per Jaar'
    fig_3 = px.bar(df_grouped_year, x='year', color=fig3_color, y=temp_unit, title=fig3_title)
    fig_3.update_traces(name='Temperatuur', showlegend=True)
    fig_3.update_layout(
        xaxis_title='Tijd',
        yaxis_title=f'{unit}',
        width=900,
        height=500,
        barmode='overlay',
        yaxis=dict(range=y_range),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # show ticks for each year, else it will divide the x-axis in parts for only one data point for x-axis
    year_ticks = list(range(df['year'].min(), df['year'].max() + 1)) if all_years else [year]

    fig_3.update_xaxes(
    tickvals=year_ticks,  # Set the tick values to the specified years
    ticktext=[str(year) for year in year_ticks]  # Convert year ticks to strings for labeling
)
    if show_trendline:
        X = np.arange(len(df_grouped_year)).reshape(-1, 1)
        Y = df_grouped_year[temp_unit].values
        reg = LinearRegression().fit(X, Y)
        df_grouped_year['bestfit'] = reg.predict(X)
        fig_3.add_scatter(x=df_grouped_year['year'], y=df_grouped_year['bestfit'], mode='lines', name='Trendline', line=dict(color='yellow'))

    if show_feels_like:
        fig_3.add_bar(x=df_grouped_year['year'], y=df_grouped_year['feels_like'], opacity=0.5, name='Feels Like', marker_color='red')

    if show_dew_point:
        fig_3.add_bar(x=df_grouped_year['year'], y=df_grouped_year['dew_point'], opacity=0.5, name='Dew Point', marker_color='green')

    st.header('Line Plot')
    st.write("""
De eerste grafiek toont de temperatuurontwikkeling in Amsterdam over de jaren heen. Hierbij is duidelijk te zien dat de temperatuur in de zomermaanden hoger ligt dan in de wintermaanden. Ook is er een stijgende trend zichtbaar in de temperatuur.
"""
    )
             
    st.plotly_chart(fig, use_container_width=True)
    
    st.header('Bar Plots')
    st.write("""
De eerste grafiek toont de gemiddelde maandelijkse temperaturen in Amsterdam, waarbij duidelijk de verschillen zichtbaar zijn. In de wintermaanden, in de eerste maanden ligt de gemiddelde tempratuur lager. Daarna stijgt de tempratuur snel. Als je naar het eind van de eerste grafiek kijkt dan koelt het weer af richting de herfst en winter.
De tweede grafiek geeft de gemiddelde temperatuur per jaar weer. Hiermee kun je de jaren met elkaar vergelijken.
"""
    )
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(fig_2, use_container_width=True)
    
    with col2:
        st.plotly_chart(fig_3, use_container_width=True)
    
    st.write('---')    
    
    st.header('Slide door de maanden van het jaar')
    st.write('''
Deze grafiek maakt het mogelijk om een specifieke maand van het jaar te selecteren, waarna de temperatuurontwikkeling gedurende die maand overzichtelijk wordt weergegeven. Hiermee kun je eenvoudig inzicht krijgen in de temperatuurfluctuaties binnen de gekozen periode.
'''
    )
    month_fig4 = st.slider('Selecteer een Maand', min_value=1, max_value=12, value=1, step=1, format="%d")

    # based on the selected year, and month, create a graph that shows the temperature of that month
    df_filtered_month = df_filtered[(df_filtered['year'] == year) & (df_filtered['month'] == month_fig4)]
    fig_4 = px.line(df_filtered_month, x='dt', y=temp_unit, title=f'Temperatuur in {year}-{month_fig4}')
    fig_4.update_traces(name='Temperatuur', showlegend=True)
    fig_4.update_layout(
        xaxis_title='Tijd',
        yaxis_title=f'{unit}',
        width=900,
        height=500,
        yaxis=dict(range=y_range),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    if show_feels_like:
        fig_4.add_scatter(x=df_filtered_month['dt'], y=df_filtered_month['feels_like'], opacity=0.5, mode='lines', name='Feels Like',
                        line=dict(color='red'))	
    if show_dew_point:
        fig_4.add_scatter(x=df_filtered_month['dt'], y=df_filtered_month['dew_point'], opacity=0.5, mode='lines', name='Dew Point',
                        line=dict(color='green'))
    
    if show_trendline:
        X = np.arange(len(df_filtered_month)).reshape(-1, 1)
        Y = df_filtered_month[temp_unit].values
        reg = LinearRegression().fit(X, Y)
        df_filtered_month['bestfit'] = reg.predict(X)
        fig_4.add_scatter(x=df_filtered_month['dt'], y=df_filtered_month['bestfit'], mode='lines', name='Trendline', line=dict(color='yellow'))

    st.plotly_chart(fig_4, use_container_width=True)




with st.sidebar.expander("Zie Locatie"):
    # Add a Folium map with a marker for Amsterdam (only create it once)
    if "map" not in st.session_state:
        # Get lat/lon values
        lat = df.head(1)['lat'].values[0]
        long = df.head(1)['lon'].values[0]

        # Create a Folium map centered on the location
        m = folium.Map(location=[lat, long], zoom_start=7)

        # Add a marker with a popup
        folium.Marker([lat, long], popup="Amsterdam").add_to(m)

        # Save the map in session state
        st.session_state["map"] = m

    st.markdown("**Gemeten Locatie**")
    folium_static(st.session_state["map"], width=250, height=150)


if st.session_state.visibility == "api":
    # Create a title for the API section
    st.title('API')

    # create a alinea with text 
    st.write('We hebben gebruik gemaakt van de OpenWeather api. Het was ons doel om inzicht te krijgen van de temperatuur in Amsterdam. Hieronder staat de code van de api.')

    st.markdown('**Dit is een functie om een lijst met unix timestamps te maken op basis van een lijst met jaren**')
    code = """
def convert_date_to_timestamp(date_string):
    timestamp = int(time.mktime(time.strptime(date_string + " 16:00:00", "%d-%m-%Y %H:%M:%S")))
    return timestamp

years = [2017, 2018, 2019, 2020, 2021, 2022]
date_strings = []

for year in years:
    # Loop through each month and day
    for i in range(1, 13):
        # Get the number of days in the current month
        days_in_month = calendar.monthrange(year, i)[1]

        # Loop through the days of the current month and add the converted date to the list
        for j in range(1, days_in_month + 1):
            date_strings.append(convert_date_to_timestamp(f"{j:02d}-{i:02d}-{year}"))
"""
    st.code(code, language='python')

    st.header('**Api tokens**')
    code = """
API_key = 'Hier de api key'
Amsterdam_lat = 52.377956
Amsterdam_lon = 4.897070
"""
    st.code(code, language='python')

    st.header('**Dit is de code om de data te krijgen en op te slaan in een csv bestand**')
    code = """
# Loop through each date and request the weather data
for timestamp in date_strings:
   url = f"https://api.openweathermap.org/data/3.0/onecall/timemachine?lat={Amsterdam_lat}&lon={Amsterdam_lon}&dt={timestamp}&appid={API_key}"
   
   with requests.get(url) as response:
      data = response.json()
      df = pd.DataFrame(data)


   # Check if the file exists
      file_exists = os.path.exists('weather.csv')

      # Write to CSV: Only include the header the first time (if the file doesn't exist)
      df.to_csv('weather.csv', mode='a', header=not file_exists, index=False)
"""
    st.code(code, language='python')

    st.header('**Hieronder zie je de eerste 3 rijen van de data**')
    st.dataframe(df_raw.head(3))
    st.write('zoals je ziet is de data kolom nog opgekropt. Dit hebben we opgelost door het op te schonen')
    st.header('**Data opschonen en naar een nieuwe csv schrijven**')
    code = """
# Parse the 'data' column (stringified dictionary) back into a Python dictionary 
df['data'] = df['data'].apply(ast.literal_eval)

# Flatten the 'data' column into multiple columns
data_columns = df['data'].apply(pd.Series)

# Extract and flatten 'weather' from the nested dictionary inside 'data'
data_columns['weather_id'] = data_columns['weather'].apply(lambda x: x[0]['id'] if isinstance(x, list) and len(x) > 0 else None)
data_columns['weather_main'] = data_columns['weather'].apply(lambda x: x[0]['main'] if isinstance(x, list) and len(x) > 0 else None)
data_columns['weather_description'] = data_columns['weather'].apply(lambda x: x[0]['description'] if isinstance(x, list) and len(x) > 0 else None)
data_columns['weather_icon'] = data_columns['weather'].apply(lambda x: x[0]['icon'] if isinstance(x, list) and len(x) > 0 else None)

# Drop the original 'weather' column as it's no longer needed
data_columns = data_columns.drop(columns=['weather'])

# Combine the original DataFrame with the newly created columns
df = pd.concat([df.drop(columns=['data']), data_columns], axis=1)


# before we save the cleaned data to a new CSV file, we need to convert the 'dt' column to a human-readable date
df['dt'] = pd.to_datetime(df['dt'], unit='s')


# Convert 'dt' column to datetime
df['dt'] = pd.to_datetime(df['dt'])

# add column year and month and week to the dataframe
df['month'] = df['dt'].dt.month
df['year'] = df['dt'].dt.year
df['week'] = df['dt'].dt.isocalendar().week

# add fahrenheit, and celsius columns, and kelvin to the dataframe
df['fahrenheit'] = df['temp'] * 9/5 - 459.67
df['celsius'] = df['temp'] - 273.15
df['kelvin'] = df['temp']

# Save the cleaned data to a new CSV file
df.to_csv('weather_cleaned.csv', index=False)
"""
    st.code(code, language='python')
    st.header('**Hieronder zie je de eerste 3 rijen van de opgeschoonde data**')
    st.dataframe(df.head(3))
    st.write('---')

    # add url to the api
    st.header('**Betrouwbaarheid - OpenWeather API**')
    st.write('''
Openweather is de eerste api die te voorschijn komt als je zoekt naar een weer api. Het is een gratis api en heeft een goede documentatie.
''')
    st.write('''
https://openweathermap.org/accuracy-and-quality
'''
)

    
