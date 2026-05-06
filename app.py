from flask import Flask, render_template, request, send_file, jsonify
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server compatibility
import matplotlib.pyplot as plt
import io
import base64
import json
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

def generate_dynamic_forecast(data, forecast_period):
    """
    Generate ARIMA forecast with automatic parameter selection

    Parameters:
    data (pandas.Series): Time series data with datetime index
    forecast_period (int): Number of periods to forecast

    Returns:
    numpy.array: Forecasted values
    """
    # Check if we have enough data
    if len(data) < 12:
        # If we have very limited data, use simple moving average
        mean_value = data.mean()
        return np.array([mean_value] * forecast_period)
    
    # Check for stationarity
    adf_result = adfuller(data, autolag='AIC')
    p_value = adf_result[1]
    
    # If not stationary (p-value > 0.05), take first difference
    d = 1 if p_value > 0.05 else 0
    
    # Try different ARIMA models with various p, q values
    best_aic = float('inf')
    best_model = None
    best_params = None
    
    # Grid search for best p, q parameters
    for p in range(0, 3):
        for q in range(0, 3):
            try:
                model = ARIMA(data, order=(p, d, q))
                model_fit = model.fit()
                if model_fit.aic < best_aic:
                    best_aic = model_fit.aic
                    best_model = model_fit
                    best_params = (p, d, q)
            except:
                continue
    
    # If we couldn't find a good ARIMA model, try SARIMAX with seasonality
    if best_model is None:
        try:
            # Try with monthly seasonality (12)
            model = SARIMAX(data, order=(1, d, 1), seasonal_order=(1, 0, 1, 12))
            best_model = model.fit(disp=False)
            best_params = "SARIMAX"
        except:
            # Fallback to simple exponential smoothing
            alpha = 0.3
            smoothed = [data[0]]
            for i in range(1, len(data)):
                smoothed.append(alpha * data[i] + (1 - alpha) * smoothed[i-1])
            forecast = np.array([smoothed[-1]] * forecast_period)
            if len(data) >= 6:
                recent_trend = (data[-1] - data[-6]) / 6
                for i in range(forecast_period):
                    forecast[i] += recent_trend * (i + 1)
            return forecast
    
    # Generate forecast using the best model
    if best_model is not None:
        forecast = best_model.forecast(steps=forecast_period)
        # Add some randomness and a diminishing trend effect for realism
        trend = 0
        if len(data) >= 6:
            trend = (data[-1] - data[-6]) / 6
        std_dev = data.tail(12).std() * 0.2
        noise = np.random.normal(0, std_dev, forecast_period)
        for i in range(forecast_period):
            forecast[i] += noise[i] + (trend * i * 0.5)
        forecast = np.maximum(forecast, 0)
        return forecast.values
    else:
        return np.array([data.mean()] * forecast_period)

def filter_by_time_frame(df, time_frame):
    """Filter dataframe based on selected time frame"""
    today = datetime.now()
    if time_frame == 'all' or time_frame is None:
        return df
    elif time_frame == '1week':
        start_date = today - timedelta(days=7)
    elif time_frame == '1month':
        start_date = today - timedelta(days=30)
    elif time_frame == '3months':
        start_date = today - timedelta(days=90)
    elif time_frame == '6months':
        start_date = today - timedelta(days=180)
    elif time_frame == '1year':
        start_date = today - timedelta(days=365)
    else:
        return df
    return df[df['Date'] >= start_date]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Validate forecast period input
            forecast_period = request.form['forecast_period']
            if not forecast_period.isdigit() or int(forecast_period) <= 0:
                return render_template('index.html', error="Forecast period must be a positive integer.")
            forecast_period = int(forecast_period)
            
            chart_type = request.form.get('chart_type', 'line')
            time_frame = request.form.get('time_frame', 'all')

            # Process uploaded file or use sample data
            if 'data_file' in request.files and request.files['data_file'].filename != '':
                file = request.files['data_file']
                filename = file.filename.lower()
                file_bytes = file.read()
                if filename.endswith('.csv'):
                    df = pd.read_csv(io.BytesIO(file_bytes))
                else:
                    df = pd.read_excel(io.BytesIO(file_bytes), engine='openpyxl')
                required_columns = ['Date', 'Sales']
                if 'buying_price' in request.form and 'selling_price' in request.form and request.form['buying_price'] and request.form['selling_price']:
                    buying_price_column = request.form['buying_price']
                    selling_price_column = request.form['selling_price']
                    required_columns.extend([buying_price_column, selling_price_column])
                else:
                    buying_price_column = 'Buying_Price'
                    selling_price_column = 'Selling_Price'
                    if buying_price_column in df.columns and selling_price_column in df.columns:
                        required_columns.extend([buying_price_column, selling_price_column])
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    return render_template('index.html', error=f"Excel file missing required columns: {', '.join(missing_columns)}.")
                
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                if df['Date'].isnull().any():
                    return render_template('index.html', error="Some dates in the 'Date' column are invalid.")
                
                numeric_columns = ['Sales']
                if buying_price_column in df.columns and selling_price_column in df.columns:
                    numeric_columns.extend([buying_price_column, selling_price_column])
                for col in numeric_columns:
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        return render_template('index.html', error=f"'{col}' column must contain numeric values.")
                    if df[col].isnull().any():
                        return render_template('index.html', error=f"'{col}' column contains missing values.")
                
                df = df.sort_values('Date')
                if buying_price_column in df.columns and selling_price_column in df.columns:
                    df['Profit'] = df[selling_price_column] - df[buying_price_column]
                    df['Profit_Margin'] = (df['Profit'] / df['Sales']) * 100
                
                df = filter_by_time_frame(df, time_frame)
                if len(df) == 0:
                    return render_template('index.html', error=f"No data available for selected time frame.")
                
                df['Date'] = df['Date'].dt.to_period('M').dt.to_timestamp('M')
                if df['Date'].dt.day.nunique() > 1:
                    data = df.set_index('Date')['Sales'].resample('M').sum()
                else:
                    data = pd.Series(df['Sales'].values, index=df['Date'])
                
                profit_data = None
                if 'Profit' in df.columns:
                    if df['Date'].dt.day.nunique() > 1:
                        profit_data = df.set_index('Date')['Profit'].resample('M').sum()
                    else:
                        profit_data = pd.Series(df['Profit'].values, index=df['Date'])
            else:
                np.random.seed(42)
                dates = pd.date_range(start='2024-01-01', periods=60, freq='M')
                trend = np.linspace(1000, 2000, 60)
                seasonal = 200 * np.sin(np.linspace(0, 10*np.pi, 60))
                noise = np.random.normal(0, 50, 60)
                sales = trend + seasonal + noise
                buying_prices = sales * 0.6 + np.random.normal(0, 30, 60)
                selling_prices = sales * 1.1 + np.random.normal(0, 40, 60)
                profits = selling_prices - buying_prices
                df = pd.DataFrame({
                    'Date': dates,
                    'Sales': sales,
                    'Buying_Price': buying_prices,
                    'Selling_Price': selling_prices,
                    'Profit': profits
                })
                df = filter_by_time_frame(df, time_frame)
                data = pd.Series(df['Sales'].values, index=df['Date'])
                profit_data = pd.Series(df['Profit'].values, index=df['Date'])

            forecast = generate_dynamic_forecast(data, forecast_period)
            forecast_index = pd.date_range(start=data.index[-1] + pd.offsets.MonthEnd(1),
                                           periods=forecast_period, freq='M')
            forecast_df = pd.DataFrame({'Month': forecast_index, 'Forecast': forecast})
            forecast_df['Month'] = forecast_df['Month'].dt.strftime('%b %Y')
            forecast_list = forecast_df.to_dict('records')

            fig, ax = plt.subplots(figsize=(10, 5))
            if chart_type == 'line':
                ax.plot(data.index, data, label='Historical Sales', color='blue')
                ax.plot(forecast_index, forecast, label='Forecasted Sales', color='red', linestyle='--')
                if profit_data is not None:
                    ax.plot(profit_data.index, profit_data, label='Profit', color='green')
            elif chart_type == 'bar':
                ax.bar(data.index, data, label='Historical Sales', color='blue', alpha=0.7)
                ax.bar(forecast_index, forecast, label='Forecasted Sales', color='red', alpha=0.5)
                if profit_data is not None:
                    ax2 = ax.twinx()
                    ax2.bar(profit_data.index, profit_data, label='Profit', color='green', alpha=0.3)
                    ax2.set_ylabel('Profit')
                    lines, labels = ax.get_legend_handles_labels()
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax.legend(lines + lines2, labels + labels2, loc='best')
            elif chart_type == 'pie':
                if profit_data is not None:
                    total_sales = data.sum()
                    total_profit = profit_data.sum()
                    ax.pie([total_sales, total_profit], labels=['Total Sales', 'Total Profit'], 
                           autopct='%1.1f%%', colors=['blue', 'green'])
                else:
                    monthly_data = data.resample('M').sum() if not isinstance(data.index, pd.DatetimeIndex) else data
                    ax.pie(monthly_data, labels=[d.strftime('%b %Y') for d in monthly_data.index], 
                           autopct='%1.1f%%')
                ax.axis('equal')
            
            if chart_type != 'pie':
                ax.set_title(f'Sales and Forecast ({time_frame})')
                ax.set_xlabel('Date')
                ax.set_ylabel('Sales')
                ax.legend()
                ax.grid(True)
                plt.xticks(rotation=45)
                fig.tight_layout()
            else:
                ax.set_title(f'Sales Distribution ({time_frame})')
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
            plt.close(fig)
            
            download_data = df.copy()
            if 'Profit' not in download_data.columns and 'Buying_Price' in download_data.columns and 'Selling_Price' in download_data.columns:
                download_data['Profit'] = download_data['Selling_Price'] - download_data['Buying_Price']
            download_data['Date'] = download_data['Date'].dt.strftime('%Y-%m-%d')
            download_json = download_data.to_json(orient='records', date_format='iso')

            return render_template('result.html', 
                                  forecast_list=forecast_list, 
                                  plot_url=plot_url,
                                  download_data=download_json,
                                  chart_type=chart_type,
                                  time_frame=time_frame)
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            return render_template('index.html', error=f"An error occurred: {str(e)}\n\nDetails: {error_details}")
    return render_template('index.html')

@app.route('/download_sample')
def download_sample():
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=60, freq='M')
    trend = np.linspace(1000, 2000, 60)
    seasonal = 200 * np.sin(np.linspace(0, 10*np.pi, 60))
    noise = np.random.normal(0, 50, 60)
    sales = trend + seasonal + noise
    buying_prices = sales * 0.6 + np.random.normal(0, 30, 60)
    selling_prices = sales * 1.1 + np.random.normal(0, 40, 60)
    profits = selling_prices - buying_prices
    df_sample = pd.DataFrame({
        'Date': dates,
        'Sales': sales,
        'Buying_Price': buying_prices,
        'Selling_Price': selling_prices,
        'Profit': profits
    })
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_sample.to_excel(writer, index=False, sheet_name='Sheet1')
    output.seek(0)
    return send_file(output,
                     as_attachment=True,
                     download_name="sample_data.xlsx",
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

@app.route('/download_forecast_excel', methods=['POST'])
def download_forecast_excel():
    data_str = request.form.get('forecast_data', '')
    try:
        data_list = json.loads(data_str)
    except Exception as e:
        return f"Error generating download: {e}"
    df = pd.DataFrame(data_list)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Forecast')
    output.seek(0)
    return send_file(
        output,
        as_attachment=True,
        download_name='forecast.xlsx',
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

@app.route('/download_data_excel', methods=['POST'])
def download_data_excel():
    data_str = request.form.get('data_json', '')
    try:
        data_list = json.loads(data_str)
        df = pd.DataFrame(data_list)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Sales Data')
        output.seek(0)
        return send_file(
            output,
            as_attachment=True,
            download_name='sales_data.xlsx',
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    except Exception as e:
        return f"Error generating download: {e}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
# if __name__ == '__main__':
#     app.run(debug=True)