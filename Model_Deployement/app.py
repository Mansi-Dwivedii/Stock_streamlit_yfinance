#from pydoc import describe
from codecs import ignore_errors
from distutils.core import setup
from doctest import REPORT_ONLY_FIRST_FAILURE
from fileinput import filename
from tabnanny import verbose
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st
import yfinance as yf
from datetime import date
from datetime import timedelta
import datetime as dt

from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

from statsmodels.tsa.seasonal import seasonal_decompose

start = '2015-01-01'
end = date.today()
#:bar_chart:

st.title('Stock Price Prediction :roller_coaster:')

user_input = st.text_input('Enter Stock Ticker', 'RELIANCE.NS')

stock_info = yf.Ticker(user_input)
df = stock_info.history(start=start, end=end)
df.head(10)
market_price = df['Close'].tail(1)

#df = data.DataReader(user_input, 'yahoo', start, end)

#Describing Data
st.write('Data From ', start, ' to ',end)
st.write('Market Price : ', market_price)
st.write(df.tail(10))

#EDA
st.subheader('Description of the Data')
st.write(df.describe())

#Visualization 
import joblib

st.subheader('Exploratory Data Analysis')
choice = st.radio("Navigate", ["Moving Average", "Profiling", "Seasonality and Trends"])
st.info("This Navigator allows you to have a detailed view of the Trends and Seasonality")

if choice == "Profiling":
    st.header("Automated Exploratory Data Analysis")
    profile_report = ProfileReport(df)
    st_profile_report(profile_report)

elif choice == "Moving Average":
    st.subheader("")
    my_range = range(20,201)
    number = st.select_slider("Select Moving Average", options=my_range)
    st.write("Selected Moving Average is ", number)
    st.subheader('Closing Price vs Time Chart with selected MA')
    maslider = df.Close.rolling(number).mean()
    fig = plt.figure(figsize=(12,7))
    plt.plot(maslider)
    plt.plot(df.Close)
    #ax = df['Close'].plot()
    #maslider.plot(ax=ax)
    st.pyplot(fig)

elif choice == "Seasonality and Trends":
    st.header("Seasonality and Trends")
    plt.rc('figure',figsize=(14,8))
    plt.rc('font',size=15)
    target = st.selectbox("Select Your Column", df.columns)
    #result = seasonal_decompose(x=df[target], period=2)
    result = seasonal_decompose(x=df[target], model='additive', period=100)
    fig1 = result.plot()
    st.pyplot(fig1)

else: 
    pass


#Splitting Data in training and testing
newData = pd.DataFrame(df['Close'])
split = len(newData) - 100

# normalize the new dataset
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))
finalData = newData.values

trainData = finalData[0:split, :]
validData = finalData[split:, :]

scaledData = scaler.fit_transform(newData)

#Load LSTM model
model = load_model("keras_model.h5")

#Testing part
xTestData = []

inputsData = newData[len(newData) - len(validData) - 60:].values
inputsData = inputsData.reshape(-1,1)
inputsData = scaler.transform(inputsData)

for i in range(60, inputsData.shape[0]):
    xTestData.append(inputsData[i-60:i, 0])

xTestData = np.array(xTestData)
xTestData = np.reshape(xTestData, (xTestData.shape[0], xTestData.shape[1], 1))

predictedClosingPrice = model.predict(xTestData)
predictedClosingPrice = scaler.inverse_transform(predictedClosingPrice)


# visualize the results
trainData = newData[:split]
validData = newData[split:]

validData['Predictions'] = predictedClosingPrice


st.subheader("Predictions vs Original")
fig2= plt.figure(figsize = (12,7))
plt.xlabel('Date')
plt.ylabel('Close Price (Rs.)')

#trainData['Close'].plot(legend = True, color = 'blue', label = 'Train Data')
validData['Close'].plot(legend = True, color = 'green', label = 'Actual Data')
validData['Predictions'].plot(legend = True, grid = True, color = 'purple', label = 'Predicted Data', title = 'stockTitle')
st.pyplot(fig2)



#Forecasting using calender

st.subheader('Stock Price Prediction by Date')

df1=df.reset_index()['Close']
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))
#datemax="24/06/2022"
datemax=dt.datetime.strftime(dt.datetime.now() - timedelta(1), "%d/%m/%Y")
datemax =dt.datetime.strptime(datemax,"%d/%m/%Y")
x_input=df1[:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

date1 = st.date_input("Select Future Date for Predictions")
result = st.button("Predict")

if result:
        from datetime import datetime
        my_time =datetime.min.time()
        date1 = datetime.combine(date1, my_time)
        nDay = date1-datemax
        nDay = nDay.days

        date_rng = pd.date_range(start=datemax, end=date1, freq='D')
        date_rng = date_rng[1:date_rng.size]
        lst_output = []
        n_steps = x_input.shape[1]
        i = 0
       
        while(i<=nDay):

            if(len(temp_input)>n_steps):
                x_input = np.array(temp_input[1:])
                print("{} day input {}".format(i,x_input))
                x_input = x_input.reshape(1, -1)
                x_input = x_input.reshape((1, n_steps, 1))

                yhat = model.predict(x_input, verbose=0)
                print("{} day input {}".format(i,yhat))
                temp_input.extend(yhat[0].tolist())
                temp_input=temp_input[1:]
                lst_output.extend(yhat.tolist())
                i = i + 1

            else:
                x_input = x_input.reshape((1, n_steps, 1))
                yhat = model.predict(x_input, verbose=0)
                print(yhat[0])
                temp_input.extend(yhat[0].tolist())
                print(len(temp_input))
                lst_output.extend(yhat.tolist())
                i = i+1
        res = scaler.inverse_transform(lst_output)

        output = res[nDay]
        st.write("*Predicted Price for Date :*", date1, "*is*", np.round(output[0], 2))
        st.success('The Price is {}'.format(np.round(output[0], 2)))

        predictions=res[res.size-nDay:res.size]
        print(predictions.shape)
        predictions=predictions.ravel()
        print(type(predictions))
        print(date_rng)
        print(predictions)
        print(date_rng.shape)

        @st.cache_data
        def convert_df(df):
            return df.to_csv().encode('utf-8')
        df = pd.DataFrame(data = date_rng)
        df['Predictions'] = predictions.tolist()
        df.columns =['Date','Close']
        st.write(df)
        csv = convert_df(df)
        st.download_button(
            "Press to Download",
            csv,
            "file.csv",
            "text/csv",
            key='download-csv'
        )

        #Visualization
        fig3 =plt.figure(figsize=(12,7))
        xpoints = date_rng
        ypoints =predictions
        plt.xlabel('Date')
        plt.ylabel('Close Price (Rs.)')
        plt.xticks(rotation = 90)
        df.set_index('Date', inplace=True)
        #df.plot()
        df['Close'].plot(legend = True, grid = True, color = 'purple', label = 'Predicted Data', title = 'stockTitle')
        st.pyplot(fig3)


        

    


       
