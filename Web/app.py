from flask import Flask, jsonify, request, render_template
import pandas as pd
import json
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.utils import PlotlyJSONEncoder
import json

app = Flask(__name__)

# Assuming your CSV files have columns named 'Date', 'Open', 'High', 'Low', 'Close', 'Volume'
def read_stock_data(symbol, start_date, end_date):
    df = pd.read_csv(f'data/{symbol}.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
    df['Date'] = df['Date'].dt.date 
    return df.loc[mask]

def generate_candlestick_plot(df):
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=('Candlestick', 'Volume'), 
                        specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Candlestick(
            x=df['Date'],
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            increasing_line_color='green',
            decreasing_line_color='red',
            name='Candlestick'
        ), secondary_y=False
    )

    fig.add_trace(
        go.Bar(x=df['Date'], y=df['Volume'], name='Volume', marker_color='lightsalmon', opacity=0.6),
        secondary_y=True
    )

    buys = df[df['Buys'] != 0]
    fig.add_trace(
        go.Scatter(
            x=buys['Date'],
            y=buys['Buys'],
            mode='markers+text',
            name='Buy Signals',
            marker=dict(color='green', size=10, symbol='triangle-up'),
            text='Buy',
            textposition='top center'
        ), secondary_y=False
    )
    sells = df[df['Sells'] != 0]
    fig.add_trace(
        go.Scatter(
            x=sells['Date'],
            y=sells['Sells'],
            mode='markers+text',
            name='Sell Signals',
            marker=dict(color='red', size=10, symbol='triangle-down'),
            text='Sell',
            textposition='bottom center'
        ), secondary_y=False
    )

    fig.update_layout(
        title='Stock Price with Buy, Sell Signals and Volume',
        xaxis_title='Date',
        yaxis_title='Price',
        legend_title='Legend',
        xaxis_rangeslider_visible=False,
        height=600,
        margin=dict(l=20, r=20, t=40, b=20),
        template='plotly_white'
    )

    fig.update_yaxes(title_text="Price", secondary_y=False)
    fig.update_yaxes(title_text="Volume", secondary_y=True)
    
    fig.update_xaxes(
        type='category',
        categoryorder='category ascending',
        rangebreaks=[
            dict(bounds=['sat', 'mon'])
        ],
        tickangle=-45,  
        tickformat='%d-%b-%Y', 
        nticks=20
    )
    return json.dumps(fig, cls=PlotlyJSONEncoder)

def generate_line_plot(df):
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.3, specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Closing Price'),
        secondary_y=False,
    )

    fig.add_trace(
        go.Bar(x=df['Date'], y=df['Volume'], name='Volume', marker_color='lightsalmon', opacity=0.6),
        secondary_y=True,
    )

    buys = df[df['Buys'] != 0]
    if not buys.empty:
        fig.add_trace(
            go.Scatter(
                x=buys['Date'], y=buys['Buys'], 
                mode='markers+text', name='Buy Signals',
                marker=dict(color='green', size=10, symbol='triangle-up'),
                text="Buy",
                textposition="top center"
            ),
            secondary_y=False,
        )
    sells = df[df['Sells'] != 0]
    if not sells.empty:
        fig.add_trace(
            go.Scatter(
                x=sells['Date'], y=sells['Sells'], 
                mode='markers+text', name='Sell Signals',
                marker=dict(color='red', size=10, symbol='triangle-down'),
                text="Sell",
                textposition="bottom center"
            ),
            secondary_y=False,
        )
    fig.update_layout(
        title='Stock Closing Price, Volume, and Trade Signals',
        xaxis_title='Date',
        yaxis_title='Closing Price',
        yaxis2_title='Volume',
        legend_title='Legend',
        template='plotly_white',
        height=600,
        margin=dict(l=20, r=20, t=40, b=20),
    )

    fig.update_yaxes(title_text="Closing Price", secondary_y=False)
    fig.update_yaxes(title_text="Volume", secondary_y=True, showgrid=False)
    fig.update_traces(hoverinfo="x+y+name")
    fig.update_xaxes(
        type='category',
        categoryorder='category ascending',
        rangebreaks=[
            dict(bounds=['sat', 'mon'])
        ],
        tickangle=-45,  
        tickformat='%d-%b-%Y', 
        nticks=20
    )

    return json.dumps(fig, cls=PlotlyJSONEncoder)


@app.route('/')
def index():
    # Assuming you have a list of stock symbols
    stock_symbols = ['RIVN', 'BB', 'SOFI', 'GME', 'AMC', 'PLTR', 'TSLA', 'AAPL', 'MSFT', 'AMZN', 'GOOG', 'AMD', 'NVDA']  # Add your own symbols
    return render_template('index.html', stock_symbols=stock_symbols)

@app.route('/get_stock_data', methods=['POST'])
def get_stock_data():
    stock_symbol = request.form['stock_symbol']
    start_date = request.form['start_date']
    end_date = request.form['end_date']
    plot_type = request.form['plot_type']  # This will be either 'candlestick' or 'line'

    df = read_stock_data(stock_symbol, start_date, end_date)

    if plot_type == 'candlestick':
        plot = generate_candlestick_plot(df)
    else:
        plot = generate_line_plot(df)

    return jsonify({'plot': plot})

if __name__ == '__main__':
    app.run(debug=True)
