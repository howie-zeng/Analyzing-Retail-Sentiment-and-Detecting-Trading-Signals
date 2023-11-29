from flask import Flask, render_template, request, jsonify
import pandas as pd

app = Flask(__name__)

# Load your stock data
file_path = 'UI\data\AAPL_2010-01-01_2023-01-01.csv'
stock_data = pd.read_csv(file_path)
print(stock_data)
stock_data['Date'] = pd.to_datetime(stock_data['Date'])

@app.route('/')
def index():
    return render_template('index.html')  # This will render an HTML template

@app.route('/filter', methods=['POST'])
def filter_data():
    # Extract filter values from the request
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')
    min_price = float(request.form.get('min_price'))
    max_price = float(request.form.get('max_price'))
    min_volume = float(request.form.get('min_volume'))
    max_volume = float(request.form.get('max_volume'))

    # Filter the data
    filtered_data = stock_data[
        (stock_data['Date'] >= pd.to_datetime(start_date)) & 
        (stock_data['Date'] <= pd.to_datetime(end_date)) &
        (stock_data['Close'] >= min_price) & 
        (stock_data['Close'] <= max_price) &
        (stock_data['Volume'] >= min_volume) & 
        (stock_data['Volume'] <= max_volume)
    ]

    # Convert filtered data to JSON or HTML format to send back to the frontend
    # ...

    return jsonify(filtered_data.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
