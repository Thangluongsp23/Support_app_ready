
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error,mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.tsaplots import month_plot, quarter_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import ParameterGrid
import warnings
import holidays
from prophet import Prophet
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
#
# import subprocess
#
# st.write("==== Installed packages (pip list) ====")
# installed = subprocess.getoutput("pip list")
# st.code(installed)
#
# # Continue with the rest of your imports...
# from PIL import Image
# import plotly
# st.write("Plotly version:", plotly.__version__)
# st.set_page_config(layout="wide")
# max_width = 400  # your desired pixel width

image = Image.open('/Users/vanthangnguyen/Documents/ML/support_picture.png')
col1, col2 = st.columns([0.6, 0.4])
with col1:
    st.write("""
    # Support Workload Prediction App

    This app predicts the **daily number of requests** in support department!

    Data obtained from the [dashboard](https://binomo2.zendesk.com/explore/studio#/dashboards/C5D158FFA56C0229E0A6066C9589B7C9220351C2277CAAFB25656C2828322F46) by Wiam A .
    """)
with col2:
    st.image(image, width=600)
# st.markdown(
#     """
#     <style>
#     .main {
#     background-color: #98FB98;}
#     </style>
#     """, unsafe_allow_html=True)

st.sidebar.header('Input Data')
st.sidebar.markdown("""
[Example CSV Tickets input file](https://github.com/Thangluongsp23/Support_project/blob/main/Tickets.csv)\n
[Example CSV Chat input file](https://github.com/Thangluongsp23/Support_project/blob/main/Chat.csv)\n
[Example Promotion CSV input file](https://github.com/Thangluongsp23/Support_project/blob/main/Chat.csv)
""")

with st.form("Predict"):
    # Collects user input features into dataframe

    df_ticket_old = None
    df_chat_old = None
    df_promotion = None

    # File uploader for tickets
    uploaded_ticket_file = st.sidebar.file_uploader("Upload your tickets CSV file", type=["csv"], key='ticket')
    if uploaded_ticket_file is not None:
        df_ticket_old = pd.read_csv(uploaded_ticket_file)
    else:
        st.warning('Awaiting CSV ticket file data to be uploaded')

    # File uploader for chat
    uploaded_chat_file = st.sidebar.file_uploader("Upload your chat CSV file", type=["csv"], key='chat')
    if uploaded_chat_file is not None:
        df_chat_old = pd.read_csv(uploaded_chat_file)
    else:
        st.warning('Awaiting CSV chat file data to be uploaded')

    # File uploader for promotions
    uploaded_promotion_file = st.sidebar.file_uploader("Upload your promotion CSV file", type=["csv"], key='promotion')
    if uploaded_promotion_file is not None:
        df_promotion = pd.read_csv(uploaded_promotion_file)
        df_promotion = df_promotion.sort_values(by='Start_Date')
        df_promotion.drop(df_promotion.index[-1], inplace=True)
        df_promotion.reset_index(drop=True, inplace=True)
    else:
        st.warning('Awaiting CSV promotion file data to be uploaded')
    st.sidebar.header("Choose Model")  # Set a header for the sidebar
    select = st.sidebar.selectbox(
        "Choose a Model that you want to apply",
        options=["SARIMAX", "Prophet + XGBoost"],
        index=0,
    )

    # Display DataFrames only if they exist
    if df_ticket_old is not None:
        with st.expander("See full ticket data table"):
            st.write(df_ticket_old)

    if df_chat_old is not None:
        with st.expander("See full chat data table"):
            st.write(df_chat_old)

    if df_promotion is not None:
        with st.expander("See full promotion data table"):
            st.write(df_promotion)
    # Sidebar for model selection
    sub_button = st.form_submit_button('Process', type="primary")
    # Display the selected model



@st.cache_data
def process_data(df):
    df.columns = (df.columns.str.replace(' ', '_', regex=True)
                  .str.lower())
    df.dropna(inplace=True)
    return df


@st.cache_data
def create_feature(df, label=None):
    """
    Creates time series features from datetime index.
    """
    df = df.copy()
    df['date'] = df.index
    df['dayofweek'] = df['date'].dt.dayofweek
    df['weekday'] = df['date'].dt.day_name()
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.isocalendar().week
    df['date_offset'] = (df.date.dt.month * 100 + df.date.dt.day - 320) % 1300
    df['season'] = pd.cut(df['date_offset'], [0, 300, 602, 900, 1300], labels=['Spring', 'Sumer', 'Fall', 'Winter'])
    X = df[['dayofweek', 'quarter', 'month', 'year', 'dayofyear', 'dayofmonth', 'weekofyear', 'weekday', 'season']]
    if label:
        y = df[label]
        return X, y
    return X


@st.cache_data
def create_holidays(df):
    import holidays
    df2 = df.copy()
    vip_bonus_day = pd.DataFrame({'holiday': 'vip_bonus',
                                  'ds': df2[
                                      (df2.index.day == 15) | (df2.index.day == 30) | (
                                                  df2.index == '2023-02-28') | (
                                              df2.index == '2024-02-29')].index,
                                  'lower_window': -2,
                                  'upper_window': 2})

    nye = pd.DataFrame({'holiday': 'new_years',
                        'ds': pd.to_datetime(['2023-01-01', '2024-01-01']),
                        'lower_window': -3,
                        'upper_window': 3})
    in_holiday = pd.DataFrame([])

    indian_holidays = holidays.IN()  # this is a dict
    # the below is the same, but takes a string:
    indian_holidays = holidays.country_holidays('IN')  # this is a dict

    for date_, name in sorted(holidays.IN(years=[2023, 2024,2025]).items()):
        in_holiday = pd.concat([in_holiday, pd.DataFrame(
            {'ds': date_, 'holiday': "Indian National Holidays", 'lower_window': -2, 'upper_window': 1},
            index=[0])],
                               ignore_index=True)

    in_holiday['ds'] = pd.to_datetime(in_holiday['ds'], format='%Y-%m-%d', errors='ignore')
    holidays = pd.concat([in_holiday, nye, vip_bonus_day]).sort_values(by='ds')
    holidays_temp = holidays.rename(columns={'ds': 'date'})
    holidays_temp['date'] = pd.to_datetime(holidays_temp['date'])

    return holidays_temp

@st.cache_data()
def create_in_holidays(df):
    import holidays
    in_holiday = pd.DataFrame([])

    indian_holidays = holidays.IN()  # this is a dict
    # the below is the same, but takes a string:
    indian_holidays = holidays.country_holidays('IN')  # this is a dict

    for date_, name in sorted(holidays.IN(years=[2023, 2024,2025]).items()):
        in_holiday = pd.concat([in_holiday, pd.DataFrame(
            {'ds': date_, 'holiday': "Indian National Holidays", 'lower_window': -2, 'upper_window': 1},
            index=[0])],
                               ignore_index=True)

    in_holiday['ds'] = pd.to_datetime(in_holiday['ds'], format='%Y-%m-%d', errors='ignore')

    return in_holiday

@st.cache_data
def prepare_data(df_chat_old, df_ticket_old, df_promotion):
    df_chat = process_data(df_chat_old)
    df_ticket = process_data(df_ticket_old)
    df_promotion = process_data(df_promotion)
    df_ticket.rename(columns={'ticket_created_-_date': 'date', 'ticket_created_-_hour': 'hour'}, inplace=True)
    df_chat.rename(columns={'chat_started_-_date': 'date', 'chat_started_-_hour': 'hour'}, inplace=True)
    df = pd.merge(df_chat, df_ticket, on=['date', 'hour'], how='outer', suffixes=('_chat', '_ticket'))
    df.fillna(0, inplace=True)
    df['total_requests'] = df['chats'] + df['tickets']
    df['date'] = pd.to_datetime(df['date'])
    df['date_time'] = df['date'].dt.strftime('%Y-%m-%d') + ' ' + df['hour'].astype(str) + ':00:00'
    df['date_time'] = pd.to_datetime(df['date_time'])
    df_daily = df.copy()
    df_daily = df_daily[['date', 'total_requests']]
    df_daily = df_daily.groupby(['date']).agg({'total_requests': "sum"}).reset_index()

    df_promotion.rename(columns={'start_date': 'date'}, inplace=True)
    df_promotion['date'] = pd.to_datetime(df_promotion['date'])
    merged_df = pd.merge(df_daily, df_promotion, how='left', on='date')

    merged_df_with_dummies = pd.get_dummies(merged_df, columns=['promotion'])

    # Display the first few rows of the updated DataFrame
    merged_df_with_dummies = merged_df_with_dummies.drop(columns=['end_date'])
    merged_df_with_dummies.fillna(0, inplace=True)

    merged_df_with_dummies.columns = merged_df_with_dummies.columns.str.replace(' ', '_', regex=True)
    merged_df_with_dummies.rename(columns={'promotion_Monthly_Weekend_Promotion': 'weekend_promotion',
                                           'promotion_WTC_Promotion': 'wtc_promotion'}, inplace=True)
    df_main = merged_df_with_dummies.copy()
    df = df_main.copy()
    df['weekend_promotion'] = df['weekend_promotion'].astype(int)
    df['wtc_promotion'] = df['wtc_promotion'].astype(int)
    if df['date'].duplicated().any():
        # Option 1: Aggregate data for duplicate timestamps (e.g., sum)
        # df = df.groupby('ds').sum().reset_index()
        # Replace 'y' with the relevant column you need to aggregate

        # Option 2: Remove duplicates (if they represent errors)
        df = df.drop_duplicates(subset=['date'], keep='last')  # Adjust 'keep' as needed

    # Create a DatetimeIndex with hourly frequency
    df = df.set_index('date').resample('D').asfreq()

    # Now you can access the frequency attribute

    X, y = create_feature(df, label='total_requests')
    df = pd.concat([X, y, df[['weekend_promotion', 'wtc_promotion']]], axis=1)
    df = df.rename(columns={'total_requests': 'y'})

    # Label Encoder the Season
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df['season'] = le.fit_transform(df['season'])
    holidays_temp = create_holidays(df)
    # Merge holidays and data frame together
    df = pd.merge(df, holidays_temp[['date', 'holiday']], on='date', how='left')
    if df['date'].duplicated().any():
        # Option 1: Aggregate data for duplicate timestamps (e.g., sum)
        # df = df.groupby('ds').sum().reset_index()
        # Replace 'y' with the relevant column you need to aggregate

        # Option 2: Remove duplicates (if they represent errors)
        df = df.drop_duplicates(subset=['date'], keep='first')  # Adjust 'keep' as needed

    df_with_dummies = pd.get_dummies(df, columns=['holiday'])
    df3 = df_with_dummies.copy()
    df = df3.rename(columns={'holiday_Indian National Holidays': 'indian_national_holidays'})

    df['weekend_promotion'] = df['weekend_promotion'].astype(int)
    df['wtc_promotion'] = df['wtc_promotion'].astype(int)
    df['indian_national_holidays'] = df['indian_national_holidays'].astype(int)
    df['holiday_vip_bonus'] = df['holiday_vip_bonus'].astype(int)
    df['holiday_new_years'] = df['holiday_new_years'].astype(int)
    df = df[['date', 'dayofweek', 'quarter', 'month', 'year', 'dayofyear',
             'dayofmonth', 'weekofyear', 'season', 'y',
             'weekend_promotion', 'wtc_promotion', 'indian_national_holidays',
             'holiday_vip_bonus']]
    df = df.set_index('date')
    return df

@st.cache_data()
def exploratory_data_analysis(df):
    tab1, tab2, tab3, tab4, tab5 = st.tabs(['Request chart', 'Monthly plot', 'Quarterly plot',
                                            'Autocorrelation plot', 'Partial autocorrelation'])
    with tab1:
        st.markdown("**Request chart**")
        fig, ax = plt.subplots(figsize=(12,8))
        ax.plot(df['y'])
        ax.set_title("Daily tickets")
        ax.set_xlabel("Date")
        ax.set_ylabel("Total requests")
        fig.autofmt_xdate()
        st.pyplot(fig)
    # Plotting the monthly seasonality
    with tab2:
        st.markdown("**Monthly plot**")
        fig1, ax = plt.subplots(figsize=(12, 8))
        fig1 = month_plot(df['y'].resample('M').mean(),
               ylabel='Ticket Numbers')
        ax.set_xlabel("Date")
        ax.set_ylabel("Monthly plot")
        fig1.autofmt_xdate()
        st.pyplot(fig1)
    with tab3:
        st.markdown("**Quarterly plot**")
        fig2, ax = plt.subplots(figsize=(12, 8))
        fig2 = quarter_plot(df['y'].resample('Q').mean(),
           ylabel = 'Ticket Numbers')
        ax.set_xlabel("Date")
        ax.set_ylabel("Quarterly plot")
        fig2.autofmt_xdate()
        st.pyplot(fig2)

    with tab4:
        st.markdown("**Autocorrelation plot**")
        fig3, ax = plt.subplots(figsize=(12, 8))
        plot_acf(df['y'], lags=100, ax=ax)
        fig3.autofmt_xdate()
        st.pyplot(fig3)
    with tab5:
        st.markdown("**Partial autocorrelation**")
        fig4, ax = plt.subplots(figsize=(12, 8))
        plot_pacf(df['y'], lags=100, ax=ax)
        fig4.autofmt_xdate()
        st.pyplot(fig4)

@st.cache_data()
def add_recent_lags(df):
    # Create lagged features for the last 3 days
    df['lag1'] = (df.index - pd.Timedelta('7 days')).map(df['y'].to_dict())
    df['lag2'] = (df.index - pd.Timedelta('2 days')).map(df['y'].to_dict())
    df['lag3'] = (df.index - pd.Timedelta('3 days')).map(df['y'].to_dict())
    return df


import pandas as pd

@st.cache_data()
def prepare_future_dataframe(df, future_days):
    """
    Prepares the future dataframe for prediction based on the given historical dataset.

    Args:
        df (DataFrame): Historical dataset with date-based features and target variable.
        future_days (int): Number of future days to predict.

    Returns:
        DataFrame: Future dataframe with all necessary features.
    """
    # Get the last date in the dataset
    last_date = df.index[-1]

    # Create a future date range
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days, freq='D')

    # Create a future dataframe
    future_df = pd.DataFrame(index=future_dates)
    # Generate date-based features
    future_df['dayofweek'] = future_df.index.dayofweek
    future_df['quarter'] = future_df.index.quarter
    future_df['month'] = future_df.index.month
    future_df['year'] = future_df.index.year
    future_df['dayofyear'] = future_df.index.dayofyear
    future_df['dayofmonth'] = future_df.index.day
    future_df['weekofyear'] = future_df.index.isocalendar().week
    future_df['season'] = (future_df['month'] % 12 + 3) // 3  # Calculate season

    # Add `weekend_promotion` feature (promotion occurs on the 1st of every month)
    future_df['weekend_promotion'] = (future_df['dayofmonth'] == 1).astype(int)

    # Add `wtc_promotion` feature (promotion on January 1st)
    future_df['wtc_promotion'] = ((future_df['month'] == 1) & (future_df['dayofmonth'] == 1)).astype(int)

    # Add holiday features
    holidays_df = create_holidays(future_df)  # Get future holidays
    future_df = future_df.merge(holidays_df[['date', 'holiday']], left_index=True, right_on='date', how='left')

    # Convert holiday column into one-hot encoding
    future_df = pd.get_dummies(future_df, columns=['holiday'])

    # Ensure all columns are present in the same order as historical data
    required_columns = df.columns
    for col in required_columns:
        if col not in future_df.columns:
            future_df[col] = 0  # Add missing columns as 0 (e.g., missing promotions)

    # Remove duplicates in case of errors
    if future_df.index.duplicated().any():
        future_df = future_df.drop_duplicates(subset=['date'], keep='first')

    # Set the index back to date
    future_df = future_df.set_index('date')

    return future_df

@st.cache_data()
def create_data_from_prophet(df, test_days):
    from prophet import Prophet

    df = df.reset_index()
    df = df.rename(columns={'date': 'ds'})
    best_param = {'changepoint_prior_scale': 0.5,
                  'holidays_prior_scale': 20,
                  'seasonality_mode': 'multiplicative',
                  'seasonality_prior_scale': 30}
    df_copy = df.copy()
    in_holiday = create_in_holidays(df)
    df_copy[['y']] = df_copy[['y']].apply(pd.to_numeric)

    df_copy['ds'] = pd.to_datetime(df_copy['ds'])
    train_set = df.iloc[:-test_days]
    test_set = df.iloc[-test_days:]

    m = Prophet(yearly_seasonality=True,
                weekly_seasonality=True,
                holidays=in_holiday,
                **best_param)
    m.add_regressor('holiday_vip_bonus')
    m.fit(train_set)
    future = m.make_future_dataframe(periods=len(test_set), freq='D')
    future = pd.concat([future, df.iloc[:, 2:]], axis=1)
    forecast = m.predict(future)
    prophet_variables = forecast.loc[:, ['trend', 'holiday_vip_bonus', 'weekly', 'yearly']]
    df_prophet = pd.concat([df.drop(columns=['holiday_vip_bonus']), prophet_variables], axis=1)
    return df_prophet


@st.cache_data
def update_and_prepare_new_data(df_chat_old, df_ticket_old, df_chat_new, df_ticket_new, df_promotion):
    # Concatenate old and new chat data
    # df_ticket_new = process_data(df_ticket_new)
    # df_chat_new = process_data(df_chat_new)
    df_chat_all = pd.concat([df_chat_old, df_chat_new], ignore_index=True)
    #
    # Concatenate old and new ticket data
    df_ticket_all = pd.concat([df_ticket_old, df_ticket_new], ignore_index=True)

    # Now process the merged chat and ticket data with the existing promotion data
    df_prepared = prepare_data(df_chat_all, df_ticket_all, df_promotion)
    df_prepared['weekend_promotion'] = (df_prepared['dayofmonth'] == 1).astype(int)

    # Add `wtc_promotion` feature (promotion on January 1st)
    df_prepared['wtc_promotion'] = ((df_prepared['month'] == 1) & (df_prepared['dayofmonth'] == 1)).astype(int)
    return df_prepared


# Session state
if "processed_data" not in st.session_state:
    st.session_state.processed_data = None
if "model_metrics" not in st.session_state:
    st.session_state.model_results = None
if "best_model" not in st.session_state:
    st.session_state.best_model = None
if "predictions" not in st.session_state:
    st.session_state.predictions = None
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None


st.divider()
if sub_button:
    st.session_state.processed_data = prepare_data(df_chat_old, df_ticket_old, df_promotion)
    st.success("✅ Data processing complete!")

    # Display Data
if st.session_state.processed_data is not None:
    st.subheader("Display the full Data table")
    with st.expander("Click here to see the table"):
        st.write(st.session_state.processed_data)

    st.subheader("Exploratory Data Analysis")
    exploratory_data_analysis(st.session_state.processed_data)

# Divider
st.divider()
st.title("Model Testing and Prediction")

# Form to choose between Testing & Predicting
with st.form("Test or Predict"):
    st.markdown("### Please choose the option that you want: ###")

    # Create two columns for side-by-side button placement
    col1, col2, col3 = st.columns([0.4, 0.3, 0.4])  # Adjust column width ratio if needed

    with col1:
        test_bt = st.form_submit_button("Test/Compare Models", use_container_width=True, type="secondary")  # Full-width button

    with col2:
        predict_bt = st.form_submit_button("Predict", use_container_width=True, type="secondary")  # Full-width button

    with col3:
        test_new_bt = st.form_submit_button("Test on New Data", use_container_width=True, type="secondary")  # Full-width button


# Session state
if "show_test_tabs" not in st.session_state:
    st.session_state.show_test_tabs = False
if "show_predict_results" not in st.session_state:
    st.session_state.show_predict_results = False
if "show_test_new_results" not in st.session_state:
    st.session_state.show_test_new_results = False

if test_bt:
    st.session_state.show_test_tabs = True
    st.session_state.show_predict_results = False
    st.session_state.show_test_new_results = False
if predict_bt:
    st.session_state.show_predict_results = True
    st.session_state.show_test_tabs = False
    st.session_state.show_test_new_results = False
if test_new_bt:
    st.session_state.show_predict_results = False
    st.session_state.show_test_tabs = False
    st.session_state.show_test_new_results = True

if st.session_state.show_test_tabs:
    tab_test, tab_compare = st.tabs(
        ["Model Testing", "Model Comparison"]
    )
    with tab_test:
        st.info("Running model evaluation...")
        df = st.session_state.processed_data
        if df is None:
            st.error("⚠️ Please process the data first!")
            st.stop()
        if select == 'SARIMAX':
            model_predict = st.container()
            with model_predict:
                st.subheader(f"Model prediction of the selected model: {select}")
                prediction_table = st.container()
                with prediction_table:
                    col1, col2 = st.columns([0.7, 0.3])
                    with col1:


                # Reads in saved classification model
                # Assuming the model is loaded in the original environment
                # Load the pre-trained model
                        y = df['y']
                        # exogenous variables
                        X = df.iloc[:, -1:]
                        X_future = X.copy()
                        tuned_model = SARIMAX(y,
                                              exog=X,
                                              order=(3, 1, 1),
                                              seasonal_order=(1, 0, 1, 7),
                                              enforce_stationarity=False,
                                              enforce_invertibility=False)

                        # Fit the model
                        support_model = tuned_model.fit(disp=False)
                        predictions = support_model.predict(start=0, end=len(y) - 1, exog=X_future)
                        # st.subheader('Prediction')
                     # Convert to datetime if not already
                        df_result = pd.DataFrame({
                            'Actual': y,  # Actual values
                            'Predicted': predictions  # Predicted values
                        }, index=df.index)  # Set date as the index]
                        df_result = df_result.reset_index()
                        df_result['date'] = pd.to_datetime(df_result['date'])

                        # Format the 'date' column to display only the date (YYYY-MM-DD)
                        df_result['date'] = df_result['date'].dt.strftime('%Y-%m-%d')
                        df_result = df_result.set_index("date")
                        predicted_df = df_result.iloc[-31:,:]
                        st.markdown("### **Result table:**")
                        st.dataframe(predicted_df.style.format({"Actual": "{:,.0f}", "Predicted": "{:,.2f}"}),
                                     use_container_width=True)

                        with col2:
                            st.markdown("### **Error Metrics**")
                    # Calculating the MAE, RMSE, and MAPE
                            mae = mean_absolute_error(df.y, predictions)
                            rmse = np.sqrt(mean_squared_error(df.y, predictions))
                            mape = mean_absolute_percentage_error(df.y, predictions)
                            st.divider()
                            st.markdown(f"The Mean absolute error metric **MAE** is: {mae:.2f}")
                            st.divider()
                            st.markdown(f"The Root mean squared error metric **RMSE** is: {rmse:.2f}")
                            st.divider()
                            st.markdown(f"The Mean absolut percentage error **MAPE** is: {100 * mape:.2f} %")

                            # >>> STORE METRICS FOR COMPARISON <<<
                            if "model_metrics" not in st.session_state:
                                st.session_state.model_metrics = {}
                            st.session_state.model_metrics["SARIMAX"] = {
                                "MAE": mae,
                                "RMSE": rmse,
                                "MAPE": 100*mape
                            }
                st.divider()
                predictions_chart = st.container()
                with predictions_chart:
                    fig = go.Figure()

                    # Add the "Actual" values trace
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df['y'],
                        mode='lines',
                        name='Actual',
                        line=dict(color='blue', width=2),  # Blue solid line
                        marker=dict(size=5)  # Add markers
                    ))

                    # Add the "Predicted" values trace
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=predictions,
                        mode='lines',
                        name='Predicted',
                        line=dict(color='orange', dash='dash', width=2),  # Orange dashed line
                        marker=dict(size=5)  # Add markers
                    ))

                    # Update layout for improved visuals
                    fig.update_layout(
                        title="Train and Forecast with SARIMAX",
                        xaxis=dict(
                            title="Date",
                            tickangle=-45,  # Tilt x-axis labels for readability
                            showgrid=True,
                            showline=True  # Add axis lines
                        ),
                        yaxis=dict(
                            title="Values",
                            showgrid=True,
                            showline=True  # Add axis lines
                        ),
                        template="plotly_white",  # Clean and modern theme
                        hovermode="x unified",  # Unified hover to show all values for a specific x-axis
                        legend=dict(
                            bgcolor="rgba(0,0,0,0)",  # Transparent legend background
                            font=dict(size=12)
                        )
                    )

                    # Display the chart in Streamlit
                    st.plotly_chart(fig, use_container_width=True)


        elif select == "Prophet + XGBoost":
            import xgboost as xgb
            # Prepare date for XGboost
            from prophet import Prophet
            from datetime import date

            model_predict = st.container()
            # test_days = int(st.number_input("Input how many days that you want to predict"))
            test_days = 31

            with model_predict:
                st.subheader(f"Model prediction of the selected model: {select}")
                prediction_table = st.container()
                with prediction_table:
                    col1, col2 = st.columns([0.7, 0.3])
                    with col1:
                    # Reads in saved classification model
                    # Assuming the model is loaded in the original environment
                    # Load the pre-trained model

                        df_xgb = create_data_from_prophet(df, test_days=test_days)

                        params = {'learning_rate': 0.05,
                                  # The step size for updating weights during training; smaller values ensure more gradual learning
                                  'max_depth': 5,
                                  # Maximum depth of each decision tree; controls model complexity and risk of overfitting
                                  'subsample': 0.9,
                                  # Fraction of the training data used for each tree; helps prevent overfitting
                                  'colsample_bytree': 1,
                                  # Fraction of features used for each tree; 1 means all features are considered
                                  'min_child_weight': 5,
                                  # Minimum sum of instance weights needed in a child node; prevents overfitting by controlling tree splitting
                                  'gamma': 0}
                                  # # Minimum loss reduction required to make a further partition; higher values make the model more conservative
                                  # # 'random_state': 1502,  # Random seed for reproducibility of results
                                  # 'eval_metric': 'rmse'}
                                  # # Evaluation metric; RMSE (root mean squared error) is common for regression tasks
                                  # # 'objective': 'reg:squarederror'}
                        params.update({
                            'random_state': 1502,
                            'eval_metric': 'rmse',
                            'objective': 'reg:squarederror'
                        })
                        df_xgb = df_xgb.set_index('ds')
                        df_xgb = add_recent_lags(df_xgb)
                        df_xgb = df_xgb.fillna(0)
                        # Extract the last 30 days from the dataset for evaluation
                        test_start_idx = df_xgb.index[-31]  # Start from 31st to include the last 30 days
                        test_data = df_xgb.loc[test_start_idx:]

                        # Define training data excluding the last 30 days
                        train_data = df_xgb.loc[:test_start_idx]

                        # Define features and targets
                        FEATURE = ['dayofweek', 'quarter', 'month', 'year', 'dayofyear',
                                   'dayofmonth', 'weekofyear', 'season', 'trend',
                                   'holiday_vip_bonus', 'weekly', 'yearly', 'lag1', 'lag2', 'lag3']

                        X_train = train_data[FEATURE]
                        y_train = train_data['y']
                        X_test = test_data[FEATURE]
                        y_test = test_data['y']

                        # Create XGBoost matrices
                        Train = xgb.DMatrix(X_train, label=y_train)
                        Test = xgb.DMatrix(X_test, label=y_test)

                        # Train the model with the best parameters
                        model = xgb.train(params=params,
                                          dtrain=Train,
                                          num_boost_round=100,
                                          evals=[(Test, 'y')],
                                          verbose_eval=False)

                        # Make predictions
                        y_pred = pd.Series(model.predict(Test), index=X_test.index, name='XGBoost')
                        results_df = pd.DataFrame({
                            "Date": y_test.index[-30:],  # Replace y_test with your actual data variable
                            "Actual": y_test.values[-30:],  # Replace y_test with your actual data variable
                            "Predicted": y_pred.values[-30:]  # Replace y_pred with your prediction results
                        })
                        # Create DataFrame with date, actual, and predicted values
                        st.markdown("### **Result table:**")
                        st.dataframe(results_df.style.format({"Actual": "{:,.0f}", "Predicted": "{:,.2f}"}),
                                     use_container_width=True)
                    with col2:
                        st.markdown("### **Error Metrics**")
                        # mae = mean_absolute_error(y_test, y_pred)
                        # rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        # mape = mean_absolute_percentage_error(y_test, y_pred)
                        # st.divider()
                        # st.markdown(f"The Mean absolute error metric **MAE** is: {mae:.2f}")
                        # st.divider()
                        # st.markdown(f"The Root mean squared error metric **RMSE** is: {rmse:.2f}")
                        # st.divider()
                        # st.markdown(f"The Mean absolut percentage error **MAPE** is: {100 * mape:.2f} %")

                        past_predictions = model.predict(Train)

                        # Compute error metrics
                        mae = mean_absolute_error(y_train, past_predictions)
                        rmse = np.sqrt(mean_squared_error(y_train, past_predictions))
                        mape = mean_absolute_percentage_error(y_train, past_predictions)

                        st.divider()
                        st.markdown(f"The Mean Absolute Error metric **MAE** is: {mae:.2f}")
                        st.divider()
                        st.markdown(f"The Root Mean Squared Error metric **RMSE** is: {rmse:.2f}")
                        st.divider()
                        st.markdown(f"The Mean Absolute Percentage Error **MAPE** is: {100 * mape:.2f} %")

                        # >>> STORE METRICS <<<
                        if "model_metrics" not in st.session_state:
                            st.session_state.model_metrics = {}
                        st.session_state.model_metrics["Prophet+XGBoost"] = {
                            "MAE": mae,
                            "RMSE": rmse,
                            "MAPE": 100 * mape
                        }
                st.divider()
                predictions_chart = st.container()
                with predictions_chart:
                    fig = go.Figure()

                    # Add traces for actual and predicted values
                    fig.add_trace(go.Scatter(
                        x=y_test.index,
                        y=y_test,
                        mode='lines',
                        name='Actual',
                        line=dict(color='blue', width=2),
                        marker=dict(size=8, symbol='circle')
                    ))

                    fig.add_trace(go.Scatter(
                        x=y_test.index,  # Use the same x-axis for predictions
                        y=y_pred,
                        mode='lines',
                        name='Predicted',
                        line=dict(color='orange', dash='dash', width=2),
                        marker=dict(size=8, symbol='x')
                    ))

                    # Improve layout
                    fig.update_layout(
                        title="Train and Forecast with Prophet+XGBoost",
                        xaxis_title="Date",
                        yaxis_title="Target Value",
                        template="plotly_dark",  # Change theme to dark mode
                        hovermode="x unified",  # Show all values when hovering over x-axis
                        plot_bgcolor="white",  # Set background to white
                        legend=dict(
                            bgcolor="rgba(0,0,0,0)",  # Transparent legend
                            font=dict(size=12)
                        )
                    )

                    # Display the interactive Plotly chart
                    st.plotly_chart(fig)


    ALL_MODELS = ["SARIMAX", "Prophet+XGBoost"]

    with tab_compare:
        st.title("Model Comparison")

        # Check if we have any metrics in session_state
        if "model_metrics" not in st.session_state or not st.session_state.model_metrics:
            st.info("No models tested yet. Please run tests in the 'Test' tab!")
        else:
            tested_models = st.session_state.model_metrics.keys()

            st.markdown("Select which models you want to compare:")
            selected_models = st.multiselect(
                "Pick Models",
                options=ALL_MODELS,
                default=list(tested_models)  # by default, select only the tested models
            )

            if not selected_models:
                st.warning("Please select at least one model.")
                st.stop()

            # We'll build a DataFrame row by row.
            rows = []
            for model_name in selected_models:
                if model_name in tested_models:
                    # Retrieve stored metrics
                    metrics = st.session_state.model_metrics[model_name]
                    # e.g. {"MAE": 123, "RMSE": 456, "MAPE": 7.89, ...}
                    row = {
                        "Model": model_name,
                        "MAE": f"{metrics['MAE']:.2f}",
                        "RMSE": f"{metrics['RMSE']:.2f}",
                        "MAPE": f"{metrics['MAPE']:.2f} %"
                    }
                else:
                    # Model wasn't tested: show "N/A" and a warning
                    st.warning(f"You haven't tested **{model_name}** yet. "
                               "Please run it in the 'Test' tab first if you want a real comparison.")
                    row = {
                        "Model": model_name,
                        "MAE": "N/A",
                        "RMSE": "N/A",
                        "MAPE": "N/A"
                    }
                rows.append(row)

            # Create a DataFrame of the selected models (both tested & untested)
            comparison_df = pd.DataFrame(rows).set_index("Model")
            st.subheader("Comparison of Selected Models")
            st.dataframe(comparison_df)

            # OPTIONAL: Pick the best model from the tested ones only
            df_tested = comparison_df[comparison_df["RMSE"] != "N/A"].copy()
            if df_tested.empty:
                st.info("None of the selected models have metrics yet.")
            else:
                # Convert RMSE to float
                df_tested["RMSE"] = df_tested["RMSE"].astype(float)
                df_tested["MAPE"] = (df_tested["MAPE"]
                                     .str.replace('%', '', regex=True)
                                     .astype(float))

                # 2. Identify the single best model by RMSE
                best_model = df_tested["RMSE"].idxmin()

                # 3. Make a bar chart for RMSE
                st.subheader("RMSE & MAPE Comparison")
                fig_bar = px.bar(
                    df_tested.reset_index(),  # ensure "Model" is a column if needed
                    x="Model",
                    y="RMSE",
                    title="RMSE by Model"
                )
                st.plotly_chart(fig_bar, use_container_width=True)

                # 4. Show the best model by RMSE
                st.success(f"**Best model by RMSE among tested ones:** {best_model}")

                # 5. Recommend all models with MAPE < 13% (i.e. 0.13 as a decimal)
                recommended_models = df_tested[df_tested["MAPE"] < 12].index.tolist()
                if recommended_models:
                    # Convert list into a readable string
                    st.success(f"**Recommended models (MAPE < 12%):** {', '.join(recommended_models)}")
                else:
                    st.warning("No model has MAPE < 13%.")

                # best_model = df_tested["RMSE"].idxmin()
                # st.subheader("RMSE &MAPE Comparison")
                # fig_bar = px.bar(
                #     comparison_df.reset_index(),
                #     x="Model",
                #     y="RMSE",
                #     title="RMSE by Model"
                # )
                # st.plotly_chart(fig_bar, use_container_width=True)
                # st.success(f"**Best model by RMSE among tested ones:** {best_model}")


if st.session_state.show_predict_results:
    future_days = st.number_input("Enter number of future days to predict:", min_value=1, max_value=60, value=30,
                                  step=1)
    st.info("Generating predictions...")
    df = st.session_state.processed_data

    if future_days:

        if df is None:
            st.error("⚠️ Please process the data first!")
            st.stop()
        if select == 'SARIMAX':
            future_df = prepare_future_dataframe(df, future_days)  # Ensure this function is defined properly
            # st.write(future_df)
            model_predict = st.container()
            with model_predict:
                st.subheader(f"Model prediction of the selected model: {select}")
                st.divider()
                prediction_table = st.container()
                with prediction_table:
                    col1, col2 = st.columns([0.7, 0.3])
                    with col1:


                # Reads in saved classification model
                # Assuming the model is loaded in the original environment
                # Load the pre-trained model
                        y = df['y']
                        # exogenous variables
                        X = df.iloc[:, -1:]
                        X_future = future_df.iloc[:, -1:]
                        tuned_model = SARIMAX(y,
                                              exog=X,
                                              order=(3, 1, 1),
                                              seasonal_order=(1, 0, 1, 7),
                                              enforce_stationarity=False,
                                              enforce_invertibility=False)

                        # Fit the model
                        support_model = tuned_model.fit(disp=False)
                        future_predictions = support_model.predict(start=len(y), end=len(y) + len(future_df) - 1, exog=X_future)
                        # st.subheader('Prediction')
                     # Convert to datetime if not already
                        future_df_result = pd.DataFrame({
                            'Predicted': future_predictions
                        }, index=future_df.index)
                        st.markdown(f"### **Future Prediction Table for {future_days} Days:**")
                        st.dataframe(future_df_result.style.format({"Predicted": "{:,.2f}"}), use_container_width=True)

                    with col2:
                        st.markdown("### **Error Metrics**")

                        # Historical Predictions for Error Metrics
                        past_predictions = support_model.predict(start=0, end=len(y) - 1, exog=X)

                        mae = mean_absolute_error(y, past_predictions)
                        rmse = np.sqrt(mean_squared_error(y, past_predictions))
                        mape = mean_absolute_percentage_error(y, past_predictions)

                        st.divider()
                        st.markdown(f"The Mean Absolute Error metric **MAE** is: {mae:.2f}")
                        st.divider()
                        st.markdown(f"The Root Mean Squared Error metric **RMSE** is: {rmse:.2f}")
                        st.divider()
                        st.markdown(f"The Mean Absolute Percentage Error **MAPE** is: {100 * mape:.2f}")
                st.divider()
                predictions_chart = st.container()
                with predictions_chart:
                    fig = go.Figure()

                    # Plot historical actual values
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df['y'],
                        mode='lines',
                        name='Actual',
                        line=dict(color='blue', width=2)
                    ))

                    # Plot historical predictions
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=past_predictions,
                        mode='lines',
                        name='Predicted (Historical)',
                        line=dict(color='orange', dash='dash', width=2)
                    ))

                    # Plot future predictions
                    fig.add_trace(go.Scatter(
                        x=future_df.index,
                        y=future_predictions,
                        mode='lines',
                        name=f'Forecast ({future_days} Days)',
                        line=dict(color='red', dash='dot', width=2)
                    ))

                    # Update layout for better visuals
                    fig.update_layout(
                        title=f"Train and Future Forecast with SARIMAX ({future_days} Days)",
                        xaxis=dict(
                            title="Date",
                            tickangle=-45,
                            showgrid=True,
                            showline=True
                        ),
                        yaxis=dict(
                            title="Values",
                            showgrid=True,
                            showline=True
                        ),
                        template="plotly_white",
                        hovermode="x unified",
                        legend=dict(
                            bgcolor="rgba(0,0,0,0)",
                            font=dict(size=12)
                        )
                    )

                    # Display the Plotly chart
                    st.plotly_chart(fig, use_container_width=True)

        elif select == "Prophet + XGBoost":
            import xgboost as xgb
            # Prepare date for XGboost
            from prophet import Prophet
            from datetime import date

            future_df = prepare_future_dataframe(df, future_days)
            model_predict = st.container()
            # test_days = int(st.number_input("Input how many days that you want to predict"))            test_days = 31
            with model_predict:
                st.subheader(f"Model prediction of the selected model: {select}")
                prediction_table = st.container()
                with prediction_table:
                    col1, col2 = st.columns([0.7, 0.3])
                    with col1:

                        # XGBoost Preparation
                        df = df.reset_index().rename(columns={'date': 'ds', 'y': 'y'})
                        future_df = future_df.reset_index().rename(columns={'date': 'ds'})

                        best_param = {
                            'changepoint_prior_scale': 0.5,
                            'holidays_prior_scale': 20,
                            'seasonality_mode': 'multiplicative',
                            'seasonality_prior_scale': 30
                        }

                        # Define and fit the Prophet model
                        m = Prophet(yearly_seasonality=True,
                                    weekly_seasonality=True,
                                    holidays=create_in_holidays(df),
                                    **best_param)

                        m.add_regressor('holiday_vip_bonus')
                        m.fit(df)

                        # Prepare future dataframe
                        future = pd.concat([df, future_df])
                        future.reset_index(drop=True, inplace=True)
                        future_regressors = future.drop(columns=['ds', 'y'])
                        # Generate future predictions
                        forecast = m.predict(future)
                        prophet_variables = forecast.loc[:, ['trend', 'holiday_vip_bonus', 'weekly', 'yearly']]
                        df_xgb = pd.concat([future.drop(columns=['holiday_vip_bonus']), prophet_variables], axis=1)
                        # Define XGBoost parameters
                        params = {
                            'learning_rate': 0.05,
                            'max_depth': 5,
                            'subsample': 0.9,
                            'colsample_bytree': 1,
                            'min_child_weight': 3,
                            'gamma': 0,
                            'random_state': 1502,
                            'eval_metric': 'rmse',
                            'objective': 'reg:squarederror'
                        }

                        # Prepare features for XGBoost
                        df_xgb = df_xgb.set_index('ds')
                        df_xgb = add_recent_lags(df_xgb)
                        df_xgb = df_xgb.fillna(0)
                        # st.write(df_xgb)
                        # Prepare training data
                        # FEATURE = [ 'trend',
                        #            'holiday_vip_bonus']
                        FEATURE = ['dayofweek', 'quarter', 'month', 'year', 'dayofyear',
                                   'dayofmonth', 'weekofyear', 'season', 'trend',
                                   'holiday_vip_bonus', 'weekly', 'yearly'] #, 'lag1', 'lag2', 'lag3'
                        cutoff_date = df.iloc[-1]['ds']# Last available date in the historical data
                        # st.write(cutoff_date)
                        # Separate training data (excluding future days)
                        X_train = df_xgb.loc[df_xgb.index <= cutoff_date, FEATURE]
                        y_train = df_xgb.loc[df_xgb.index <= cutoff_date, 'y']

                        # Separate future data (for prediction only)

                        # Train XGBoost model
                        Train = xgb.DMatrix(X_train, label=y_train)
                        model = xgb.train(params=params,
                                          dtrain=Train,
                                          num_boost_round=100,
                                          verbose_eval=False)

                        # Prepare features for future prediction
                        X_future = df_xgb.loc[df_xgb.index > cutoff_date, FEATURE]
                        Future = xgb.DMatrix(X_future)

                        # Generate future predictions
                        future_predictions = model.predict(Future)

                        # Create DataFrame with results
                        future_df_result = pd.DataFrame({'date': df_xgb.loc[df_xgb.index > cutoff_date].index,
                            'Predicted': future_predictions
                        }, index=future_df.index)
                        future_df_result = future_df_result.set_index("date")
                        st.markdown("### **Future Prediction Table:**")
                        st.dataframe(future_df_result.style.format({"Predicted": "{:,.2f}"}), use_container_width=True)

                    with col2:
                        st.markdown("### **Error Metrics**")

                        # Generate historical predictions
                        past_predictions = model.predict(Train)

                        # Compute error metrics
                        mae = mean_absolute_error(y_train, past_predictions)
                        rmse = np.sqrt(mean_squared_error(y_train, past_predictions))
                        mape = mean_absolute_percentage_error(y_train, past_predictions)

                        st.divider()
                        st.markdown(f"The Mean Absolute Error metric **MAE** is: {mae:.2f}")
                        st.divider()
                        st.markdown(f"The Root Mean Squared Error metric **RMSE** is: {rmse:.2f}")
                        st.divider()
                        st.markdown(f"The Mean Absolute Percentage Error **MAPE** is: {100 * mape:.2f} %")

                    st.divider()
                    predictions_chart = st.container()
                    with predictions_chart:
                        fig = go.Figure()

                        # **Plot Historical Actual Values**
                        fig.add_trace(go.Scatter(
                            x=df['ds'],
                            y=df['y'],
                            mode='lines',
                            name='Actual',
                            line=dict(color='blue', width=2)
                        ))

                        # **Plot Historical Predictions**
                        fig.add_trace(go.Scatter(
                            x=df['ds'],
                            y=past_predictions,
                            mode='lines',
                            name='Predicted (Historical)',
                            line=dict(color='orange', dash='dash', width=2)
                        ))

                        # **Plot Future Predictions**
                        fig.add_trace(go.Scatter(
                            x=future_df_result.index,
                            y=future_df_result['Predicted'],
                            mode='lines',
                            name='Forecast (Future)',
                            line=dict(color='red', dash='dot', width=3)
                        ))

                        # **Fix X-Axis & Y-Axis Display**
                        fig.update_layout(
                            title="Train and Future Forecast with Prophet + XGBoost",
                            xaxis=dict(
                                title="Date",
                                tickangle=-45,
                                showgrid=True,
                                showline=True
                            ),
                            yaxis=dict(
                                title="Values",
                                showgrid=True,
                                showline=True
                            ),
                            template="plotly_white",
                            hovermode="x unified",
                            legend=dict(
                                bgcolor="rgba(0,0,0,0)",
                                font=dict(size=12)
                            )
                        )

                        # Display the chart
                        st.plotly_chart(fig, use_container_width=True)


if st.session_state.show_test_new_results:

    df = st.session_state.processed_data
    chat_new = None
    ticket_new = None
    df_new = None
    st.info("New Data Performance Testing")
    st.markdown("""
        Upload the new dataset covering dates from **20.11.2024 to 11.03.2025**.
        Make sure your CSV contains a date column (named "date") and a target column (e.g., "y").
        """)
    upload_ticket_new = st.file_uploader("Upload New Ticket Dataset (CSV)", type=["csv"])
    if upload_ticket_new is not None:
        ticket_new = pd.read_csv(upload_ticket_new)
    else:
        st.warning('Awaiting CSV ticket file data to be uploaded')
    upload_chat_new = st.file_uploader("Upload New chat Dataset (CSV)", type=["csv"])
    if upload_chat_new is not None:
        chat_new = pd.read_csv(upload_chat_new)
    else:
        st.warning('Awaiting CSV chat file data to be uploaded')
    if ticket_new is not None and chat_new is not None:
        df_new = update_and_prepare_new_data(df_chat_old,df_ticket_old, chat_new,ticket_new, df_promotion)
        if "df_new " not in st.session_state:
            st.session_state.df_new = df_new
        with st.expander("See full promotion data table"):
            st.write(df_new)

        min_train_date = df.index.max()
        max_train_date = df_new.index.max()
        st.markdown("#### Select Training Data Period")

        train_end_date = st.date_input(
            "Choose Training End Date",
            value=min_train_date,
            min_value=min_train_date,
            max_value=max_train_date
        )
        st.markdown("#### Select Prediction Time Frame")
        forecast_end_date = st.date_input(
            "Choose Forecast End Date (must be on or after the training end date)",
            value=pd.to_datetime(train_end_date) + timedelta(days=30),
            min_value=train_end_date,
            max_value=max_train_date
        )
        delta = forecast_end_date - train_end_date
        future_days = int(delta.days)
        df = df_new.loc[:pd.to_datetime(train_end_date)]
        test_data = df_new.loc[pd.to_datetime(train_end_date) + timedelta(days=1): pd.to_datetime(forecast_end_date)]
        if future_days:

            if df is None:
                st.error("⚠️ Please process the data first!")
                st.stop()
            if select == 'SARIMAX':
                future_df = prepare_future_dataframe(df, future_days)  # Ensure this function is defined properly
                # st.write(future_df)
                model_predict = st.container()
                with model_predict:
                    st.subheader(f"Model prediction of the selected model: {select}")
                    st.divider()
                    prediction_table = st.container()
                    with prediction_table:
                        col1, col2 = st.columns([0.7, 0.3])
                        with col1:


                    # Reads in saved classification model
                    # Assuming the model is loaded in the original environment
                    # Load the pre-trained model
                            y = df['y']
                            # exogenous variables
                            X = df.iloc[:, -1:]
                            X_future = future_df.iloc[:, -1:]
                            tuned_model = SARIMAX(y,
                                                  exog=X,
                                                  order=(3, 1, 1),
                                                  seasonal_order=(1, 0, 1, 7),
                                                  enforce_stationarity=False,
                                                  enforce_invertibility=False)

                            # Fit the model
                            support_model = tuned_model.fit(disp=False)
                            in_sample_pred = support_model.get_prediction(
                                start=0,
                                end=len(y) - 1,
                                exog=X
                            ).predicted_mean

                            # 4. Compute residuals = actual - predicted
                            residuals = y - in_sample_pred

                            # 5. Pick residual quantiles that define "reasonable" error bands
                            #    e.g. 10th percentile (under-forecast) and 90th percentile (over-forecast).
                            lower_q = residuals.quantile(0.05)  # 10th percentile
                            upper_q = residuals.quantile(0.95)  # 90th percentile

                            # 6. Now forecast into the future:
                            future_pred_object = support_model.get_prediction(
                                start=len(y),
                                end=len(y) + len(future_df) - 1,
                                exog=X_future
                            )

                            # The main forecast
                            yhat_future = future_pred_object.predicted_mean

                            # 7. Construct "min needed" and "max needed" by adding the chosen residual offsets.
                            #    For example, if historically the model was off by ~ -30 on the low side (10th pct)
                            #    and +40 on the high side (90th pct), we can shift the forecast by those amounts.
                            #    We also clip at 0 because negative tickets/employees make no sense.
                            yhat_lower = (yhat_future + lower_q).clip(lower=0)
                            yhat_upper= (yhat_future + upper_q).clip(lower=0)

                            # 8. Build a DataFrame
                            # prediction_results = support_model.get_prediction(
                            #     start=len(y),
                            #     end=len(y) + len(future_df) - 1,
                            #     exog=X_future
                            # )
                            #
                            # # 3. Grab the predicted mean (the forecast) and standard errors
                            # predictions = prediction_results.predicted_mean
                            # predicted_se = prediction_results.se_mean  # standard errors of the forecast
                            # st.write(predicted_se )
                            # # 4. Manually compute the interval.
                            # #    For 95% confidence, z ~ 1.96; adjust as needed for different alpha
                            # z = 1.64
                            # yhat_lower = predictions - z * predicted_se
                            # yhat_upper = predictions + z * predicted_se

                            future_df_result = pd.DataFrame({
                                'Predicted': yhat_future,
                                'yhat_lower': yhat_lower,
                                'yhat_upper': yhat_upper,
                                'Actual': test_data.y
                            }, index=future_df.index)
                            st.markdown(f"### **Future Prediction Table for {future_days} Days:**")
                            st.dataframe(future_df_result.style.format({
                                "Predicted": "{:,.2f}",
                                "yhat_lower": "{:,.2f}",
                                "yhat_upper": "{:,.2f}",
                                "Actual": "{:,.0f}"
                            }), use_container_width=True)

                        with col2:
                            st.markdown("### **Error Metrics**")

                            # Historical Predictions for Error Metrics
                            past_predictions = test_data.y

                            mae = mean_absolute_error(future_df_result['Predicted'], past_predictions)
                            rmse = np.sqrt(mean_squared_error(future_df_result['Predicted'], past_predictions))
                            mape = mean_absolute_percentage_error(future_df_result['Predicted'], past_predictions)

                            st.divider()
                            st.markdown(f"The Mean Absolute Error metric **MAE** is: {mae:.2f}")
                            st.divider()
                            st.markdown(f"The Root Mean Squared Error metric **RMSE** is: {rmse:.2f}")
                            st.divider()
                            st.markdown(f"The Mean Absolute Percentage Error **MAPE** is: {100 * mape:.2f}")
                    # # st.divider()
                    predictions_chart = st.container()
                    with predictions_chart:
                        fig = go.Figure()

                        # 1) Plot historical actual values
                        fig.add_trace(go.Scatter(
                            x=df.index,
                            y=df['y'],
                            mode='lines',
                            name='history',
                            line=dict(color='blue', width=2)
                        ))
                        fig.add_trace(go.Scatter(
                            x=future_df.index,
                            y=future_df_result['Actual'],
                            mode='lines',
                            name='Actual',
                            line=dict(color='black', width=2)
                        ))

                        # 3) Plot future predictions (central forecast)
                        fig.add_trace(go.Scatter(
                            x=future_df.index,
                            y=yhat_future,
                            mode='lines',
                            name=f'Forecast ({future_days} Days)',
                            line=dict(color='green', width=2)
                        ))

                        # 4) Plot the upper bound
                        fig.add_trace(go.Scatter(
                            x=future_df.index,
                            y=yhat_upper,
                            mode='lines',
                            name='Upper Bound',
                            line=dict(color='red', width=1),
                            # No 'fill' on this trace; we fill on the next trace to create the band
                        ))

                        # 5) Plot the lower bound, using 'fill="tonexty"' to shade between lower & upper
                        fig.add_trace(go.Scatter(
                            x=future_df.index,
                            y=yhat_lower,
                            mode='lines',
                            name='Lower Bound',
                            line=dict(color='red', width=1),
                            fill='tonexty',  # fill area from this trace up to the previous trace
                            fillcolor='rgba(255,0,0,0.2)'  # light red shading; adjust opacity/color as needed
                        ))

                        # Update layout for better visuals
                        fig.update_layout(
                            title=f"Train and Future Forecast with SARIMAX ({future_days} Days)",
                            xaxis=dict(
                                title="Date",
                                tickangle=-45,
                                showgrid=True,
                                showline=True
                            ),
                            yaxis=dict(
                                title="Tickets",
                                showgrid=True,
                                showline=True
                            ),
                            template="plotly_white",
                            hovermode="x unified",
                            legend=dict(
                                bgcolor="rgba(0,0,0,0)",
                                font=dict(size=12)
                            )
                        )

                        # Display the Plotly chart
                        st.plotly_chart(fig, use_container_width=True)

            elif select == "Prophet + XGBoost":
                import xgboost as xgb
                # Prepare date for XGboost
                from prophet import Prophet
                from datetime import date

                future_df = prepare_future_dataframe(df, future_days)
                model_predict = st.container()
                # test_days = int(st.number_input("Input how many days that you want to predict"))            test_days = 31
                with model_predict:
                    st.subheader(f"Model prediction of the selected model: {select}")
                    prediction_table = st.container()
                    with prediction_table:
                        col1, col2 = st.columns([0.7, 0.3])
                        with col1:

                            # XGBoost Preparation
                            df = df.reset_index().rename(columns={'date': 'ds', 'y': 'y'})
                            future_df = future_df.reset_index().rename(columns={'date': 'ds'})

                            best_param = {
                                'changepoint_prior_scale': 0.5,
                                'holidays_prior_scale': 20,
                                'seasonality_mode': 'multiplicative',
                                'seasonality_prior_scale': 30
                            }

                            # Define and fit the Prophet model
                            m = Prophet(yearly_seasonality=True,
                                        weekly_seasonality=True,
                                        holidays=create_in_holidays(df),
                                        **best_param)

                            m.add_regressor('holiday_vip_bonus')
                            m.fit(df)

                            # Prepare future dataframe
                            future = pd.concat([df, future_df])
                            future.reset_index(drop=True, inplace=True)
                            future_regressors = future.drop(columns=['ds', 'y'])
                            # Generate future predictions
                            forecast = m.predict(future)
                            prophet_variables = forecast.loc[:, ['trend', 'holiday_vip_bonus', 'weekly', 'yearly']]
                            df_xgb = pd.concat([future.drop(columns=['holiday_vip_bonus']), prophet_variables], axis=1)
                            # Define XGBoost parameters
                            params = {
                                'learning_rate': 0.05,
                                'max_depth': 5,
                                'subsample': 0.9,
                                'colsample_bytree': 1,
                                'min_child_weight': 5,
                                'gamma': 0,
                                'random_state': 1502,
                                'eval_metric': 'rmse',
                                'objective': 'reg:squarederror'
                            }

                            # Prepare features for XGBoost
                            df_xgb = df_xgb.set_index('ds')
                            df_xgb = add_recent_lags(df_xgb)
                            df_xgb = df_xgb.fillna(0)
                            # st.write(df_xgb)
                            # Prepare training data
                            # FEATURE = [ 'trend',
                            #            'holiday_vip_bonus']
                            FEATURE = ['dayofweek', 'quarter', 'month', 'year', 'dayofyear',
                                       'dayofmonth', 'weekofyear', 'season', 'trend',
                                       'holiday_vip_bonus', 'weekly', 'yearly', 'lag1', 'lag2', 'lag3']
                            cutoff_date = df.iloc[-1]['ds']# Last available date in the historical data
                            # st.write(cutoff_date)
                            # Separate training data (excluding future days)
                            X_train = df_xgb.loc[df_xgb.index <= cutoff_date, FEATURE]
                            y_train = df_xgb.loc[df_xgb.index <= cutoff_date, 'y']

                            # Separate future data (for prediction only)

                            # Train XGBoost model
                            Train = xgb.DMatrix(X_train, label=y_train)
                            model = xgb.train(params=params,
                                              dtrain=Train,
                                              num_boost_round=100,
                                              verbose_eval=False)

                            # Prepare features for future prediction
                            X_future = df_xgb.loc[df_xgb.index > cutoff_date, FEATURE]
                            Future = xgb.DMatrix(X_future)

                            # Generate future predictions
                            future_predictions = model.predict(Future)

                            # Create DataFrame with results
                            in_sample_pred = model.predict(Train)

                            # Compute residuals
                            residuals = y_train - in_sample_pred
                            lower_q = residuals.quantile(0.05)
                            upper_q = residuals.quantile(0.95)

                            yhat_lower = future_predictions + lower_q
                            yhat_upper = future_predictions + upper_q

                            future_df_result = pd.DataFrame({
                                'Predicted': future_predictions,
                                'yhat_lower': yhat_lower,
                                'yhat_upper': yhat_upper,
                                'Actual': test_data['y'].values  # if you have a test set to compare
                            }, index=X_future.index)

                            # Then display or do whatever you need, e.g. Streamlit:
                            st.markdown(f"### **Future Prediction Table**")
                            st.dataframe(future_df_result.style.format({
                                "Predicted": "{:,.2f}",
                                "yhat_lower": "{:,.2f}",
                                "yhat_upper": "{:,.2f}",
                                "Actual": "{:,.0f}"
                            }), use_container_width=True)

                        with col2:
                            st.markdown("### **Error Metrics**")

                            # Generate historical predictions
                            past_predictions = model.predict(Train)

                            # Compute error metrics
                            mae = mean_absolute_error(future_df_result['Predicted'], future_df_result['Actual'])
                            rmse = np.sqrt(mean_squared_error(future_df_result['Predicted'], future_df_result['Actual']))
                            mape = mean_absolute_percentage_error(future_df_result['Predicted'], future_df_result['Actual'])

                            st.divider()
                            st.markdown(f"The Mean Absolute Error metric **MAE** is: {mae:.2f}")
                            st.divider()
                            st.markdown(f"The Root Mean Squared Error metric **RMSE** is: {rmse:.2f}")
                            st.divider()
                            st.markdown(f"The Mean Absolute Percentage Error **MAPE** is: {100 * mape:.2f} %")

                    predictions_chart = st.container()
                    with predictions_chart:
                        fig = go.Figure()

                        # 1) Plot historical actual values
                        fig.add_trace(go.Scatter(
                            x=X_train.index,
                            y=df['y'],
                            mode='lines',
                            name='history',
                            line=dict(color='blue', width=2)
                        ))
                        fig.add_trace(go.Scatter(
                            x=future_df_result.index,
                            y=future_df_result['Actual'],
                            mode='lines',
                            name='Actual',
                            line=dict(color='black', width=2)
                        ))

                        # 3) Plot future predictions (central forecast)
                        fig.add_trace(go.Scatter(
                            x=future_df_result.index,
                            y=future_predictions,
                            mode='lines',
                            name=f'Forecast ({future_days} Days)',
                            line=dict(color='green', width=2)
                        ))

                        # 4) Plot the upper bound
                        fig.add_trace(go.Scatter(
                            x=future_df_result.index,
                            y=yhat_upper,
                            mode='lines',
                            name='Upper Bound',
                            line=dict(color='red', width=1),
                            # No 'fill' on this trace; we fill on the next trace to create the band
                        ))

                        # 5) Plot the lower bound, using 'fill="tonexty"' to shade between lower & upper
                        fig.add_trace(go.Scatter(
                            x=future_df_result.index,
                            y=yhat_lower,
                            mode='lines',
                            name='Lower Bound',
                            line=dict(color='red', width=1),
                            fill='tonexty',  # fill area from this trace up to the previous trace
                            fillcolor='rgba(255,0,0,0.2)'  # light red shading; adjust opacity/color as needed
                        ))

                        # Update layout for better visuals
                        fig.update_layout(
                            title=f"Train and Future Forecast with SARIMAX ({future_days} Days)",
                            xaxis=dict(
                                title="Date",
                                tickangle=-45,
                                showgrid=True,
                                showline=True
                            ),
                            yaxis=dict(
                                title="Tickets",
                                showgrid=True,
                                showline=True
                            ),
                            template="plotly_white",
                            hovermode="x unified",
                            legend=dict(
                                bgcolor="rgba(0,0,0,0)",
                                font=dict(size=12)
                            )
                        )

                        # Display the Plotly chart
                        st.plotly_chart(fig, use_container_width=True)
