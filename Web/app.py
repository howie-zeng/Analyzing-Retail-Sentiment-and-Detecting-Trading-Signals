from flask import Flask, jsonify, request, render_template
import pandas as pd
import json
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.utils import PlotlyJSONEncoder

app = Flask(__name__)

# Assuming your CSV files have columns named 'Date', 'Open', 'High', 'Low', 'Close', 'Volume'
def read_stock_data(symbol, start_date, end_date):
    df = pd.read_csv(f'data/{symbol}.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    return df.loc[mask]

def generate_candlestick_plot(df):
    fig = go.Figure(
        data=[go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            increasing_line_color='green',
            decreasing_line_color='red',
            increasing_fillcolor='green',
            decreasing_fillcolor='red'
        )]
    )
    

    # Preprocess the DataFrame to ensure no missing dates within the range
    df.set_index('Date', inplace=True)
    df = df.reindex(pd.date_range(start=df.index.min(), end=df.index.max()))
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Date'}, inplace=True)
    
    # Update x-axis to category to remove gaps
    fig.update_xaxes(type='category', categoryorder='category ascending')

    # Add buy signals
    buys = df[df['Buys'] != 0]
    fig.add_trace(go.Scatter(
        x=buys['Date'],
        y=buys['Buys'],
        mode='markers',
        name='Buy Signals',
        marker=dict(
            color='green',
            size=12,
            symbol='triangle-up'
        )
    ))

    # Add sell signals
    sells = df[df['Sells'] != 0]
    fig.add_trace(go.Scatter(
        x=sells['Date'],
        y=sells['Sells'],
        mode='markers',
        name='Sell Signals',
        marker=dict(
            color='red',
            size=12,
            symbol='triangle-down'
        )
    ))

    # Improve layout
    fig.update_layout(
        title='Stock Price with Buy and Sell Signals',
        xaxis_title='Date',
        yaxis_title='Price',
        showlegend=True,
        legend_title_text='Actions',
        xaxis_rangeslider_visible=False,
        height=600,  # Makes plot larger
        margin=dict(l=20, r=20, t=40, b=20)  # Adjust margins to fit your aesthetics
    )

    # Styling
    fig.update_traces(line=dict(width=1))
    fig.layout.template = 'plotly_white'  # A light theme that may look cleaner

    return json.dumps(fig, cls=PlotlyJSONEncoder)


@app.route('/')
def index():
    # Assuming you have a list of stock symbols
    stock_symbols = ['AAPL', 'TSLA', 'GOOG', 'AMZN']  # Add your own symbols
    return render_template('index.html', stock_symbols=stock_symbols)

@app.route('/get_stock_data', methods=['POST'])
def get_stock_data():
    stock_symbol = request.form['stock_symbol']
    start_date = request.form['start_date']
    end_date = request.form['end_date']

    df = read_stock_data(stock_symbol, start_date, end_date)
    plot = generate_candlestick_plot(df)
    # For simplicity, not implementing buy/sell actions here
    return jsonify({'plot': plot})

if __name__ == '__main__':
    app.run(debug=True)
