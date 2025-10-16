import streamlit as st
import pandas as pd
import altair as alt

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Time Series Forecast Visualization")

@st.cache_data
def load_data(file_path):
    """Loads and preprocesses the time series data, handling encoding errors."""
    try:
        # 1. Try default UTF-8 encoding first
        df = pd.read_csv(file_path)
    except UnicodeDecodeError:
        try:
            # 2. If it fails, try a common alternative: 'latin-1' (Windows-1252)
            df = pd.read_csv(file_path, encoding='latin-1')
        except Exception as e:
            st.error(f"Error: Could not load the file using standard encodings. Please check the file path and encoding. Details: {e}")
            return pd.DataFrame()

    # Convert Date column to datetime objects
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# The exact file name accessible in the environment (which came from the .xlsx)
# If your local CSV file is simply named 'Prophet_All_Areas_Forecast.csv', update this variable.
FILE_NAME = "Prophet_All_Areas_Forecast.xlsx"
df = load_data(FILE_NAME)

if not df.empty:
    st.title("Time Series Forecast Visualization")

    # --- Sidebar for Single Area Selection ---
    st.sidebar.header("Filter Options")
    all_areas = sorted(df['Area'].unique())

    # Selectbox widget for a single Area
    selected_area = st.sidebar.selectbox(
        "Select an Area to Display",
        options=all_areas,
        index=0 # Default to the first area
    )

    if not selected_area:
        st.warning("Please select an Area to display the chart.")
    else:
        # Filter the DataFrame based on the single selection
        df_filtered = df[df['Area'] == selected_area].copy()

        # 1. Determine Forecast Start Date
        # Find the earliest date where 'Prophet_Forecast' is NOT NaN for the selected area
        forecast_start_date_series = df_filtered.loc[df_filtered['Prophet_Forecast'].notna(), 'Date']
        
        forecast_start_date = None
        if not forecast_start_date_series.empty:
            # Get the day *before* the first forecast point to draw the vertical line
            # The .min() finds the start of the forecast. We use the previous month's 1st day as a proxy.
            min_forecast_date = forecast_start_date_series.min()
            # Find the date immediately preceding the forecast start date in the dataset
            previous_dates = df_filtered[df_filtered['Date'] < min_forecast_date]['Date']
            if not previous_dates.empty:
                # Use the last actual/fitted data point's date as the vertical line position
                forecast_start_date = previous_dates.max()
            else:
                forecast_start_date = min_forecast_date

        # 2. Data Transformation for Altair (Melt)
        melt_cols = ['Actual', 'Prophet_Train_Fitted', 'Prophet_Forecast', 'Auto_arima_forcasted']
        df_melted = df_filtered.melt(
            id_vars=['Area', 'Date', 'Lower_95_CI', 'Upper_95_CI'],
            value_vars=melt_cols,
            var_name='Series',
            value_name='Value'
        )

        # 3. Altair Visualization
        
        # --- Base Chart ---
        base = alt.Chart(df_melted).encode(
            x=alt.X('Date', title='Date'),
            y=alt.Y('Value', title='Value'),
            tooltip=[
                alt.Tooltip('Date:T', title='Date'),
                alt.Tooltip('Series'),
                alt.Tooltip('Value:Q', format=',.2f')
            ]
        ).properties(
            title=f"Forecast Analysis for {selected_area}"
        )

        # --- Confidence Interval Band ---
        # Filter for the CI data (only where Prophet_Forecast is available)
        df_ci = df_filtered.dropna(subset=['Prophet_Forecast'])
        ci_band = alt.Chart(df_ci).mark_area(opacity=0.2).encode(
            x='Date:T',
            y=alt.Y('Lower_95_CI:Q', title='Value'),
            y2='Upper_95_CI:Q',
            color=alt.value('gray'),
            order=alt.value(1) # Draw CI band first
        )
        
        # --- Line Marks for all series ---
        lines = base.mark_line().encode(
            color=alt.Color('Series', title='Series'),
            # Use conditional stroke dash to differentiate actual/fitted from forecasts
            strokeDash=alt.condition(
                alt.FieldOneOfPredicate(field='Series', oneOf=['Prophet_Forecast', 'Auto_arima_forcasted']),
                alt.value([5, 5]),  # Dashed for forecasts
                alt.value([1, 0])   # Solid for Actual/Fitted
            ),
            order=alt.Order('Series', sort='descending') # Draw lines on top
        )
        
        # --- Vertical Line for Forecast Start ---
        if forecast_start_date is not None:
            v_line = alt.Chart(pd.DataFrame({'Forecast Start': [forecast_start_date]})).mark_rule(
                color='red', 
                size=2,
                opacity=0.7
            ).encode(
                x='Forecast Start:T',
                tooltip=[alt.Tooltip('Forecast Start:T', title='Forecast Start Date')]
            ).interactive()
            
            # Combine all layers
            chart = (ci_band + lines + v_line).interactive()
        else:
            chart = (ci_band + lines).interactive()

        st.altair_chart(chart, use_container_width=True)

        st.markdown("---")
        st.subheader("Raw Data Preview")
        st.dataframe(df_filtered)
