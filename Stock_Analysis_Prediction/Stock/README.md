## Updates/Log
[Updates/Log](#updateslog)
   - [10/02/2023](#date-10022023)
   - [10/20/2023](#date-10202023)
   - [10/24/2023](#date-10242023)
   - [Future Updates](#future-updates)

### [Date: 10/24/2023] third to forth week
#### Progress
- Changed from XGBoost Classifier to Regressor, resulting in a 10x speed increase at the cost of a 10% performance drop (probably due to randomness).
- Came up with a new evaluation metrics. The residual now seem random.
- Added a lot more indicators into the XGBoost model; the mean absolute error in percentage has been reduced to its 1/3.
- The model can now make decent detections for selling signals.
#### Next Steps
- Undertake more feature selection.
- Incorporate more advanced indicators like Williams VIX Fix and RSI Bollinger Strategy.
- Upon completion, experiment with other models such as LSM.


### [Date: 10/20/2023] second week
#### Progress
- Successfully implemented the initial version of the RNN model for sentiment analysis, achieving an accuracy of 80% for test data.
- Collected additional data to enhance the training set.
- Successfully implemented the stock preditction models, including linear regression, LSTM, ARIMA, and STL, though only linear regression has a good prediction accuracy.
- Established the extractor to count and rank all the stock names from comments data, and seletect the top 50 names from the list.
- Performed Volume, RSI, and Moving Average analysis.
- Added a `requirements.txt` file for managing project dependencies. You can install the required packages using the following commands:
  ```bash
  pip freeze > requirements.txt
  pip install -r requirements.txt
  ```
- Implemented the early version of a moving window time series model utilizing XGBoost.

#### Challenges
- The model's performance on unseen data, specifically related to stock posts, is suboptimal with an accuracy of around 60%.
- Recognized that leveraging existing datasets might be a more efficient strategy than extracting data in real-time from APIs, especially considering the rate limits associated with the latter.
- Predition models should be explored further to enhance the performance.
- Stock name extractor only considers the stock name mentioned in the comments without detailed analysis on the sentiment and focus on the stock name.

#### Next Steps
- Plan to develop a mechanism to pinpoint the specific stock that a post refers to.
- Will need to implement a method to identify the stock a post is talking about.

#### To-Dos
- Conduct a thorough EDA on the extracted stock data, focusing on identifying potential correlations with expected social media sentiment trends.
- Identify and acquire datasets containing social media posts and comments related to the meme stocks.
- Develop a strategy for cleaning and preprocessing the social media data, ensuring it is in a usable format for sentiment analysis.
- Develop a strategy to predict stock movement using stock data.
- Add market sentiment to the model predicting stock movement.

### [Date: 10/02/2023] first week
#### Progress
- Successfully extracted stock data from Yahoo Finance, focusing on popular meme stocks like GME, AMC, and BB.
- Initiated exploratory data analysis (EDA) to identify general trends and patterns within the extracted stock data, such as price fluctuations and trading volumes during specific time frames.

#### Challenges
- Encountered API rate limits while extracting data, which hindered the efficiency of the data retrieval process.
- Realized that searching and utilizing available datasets might be a more efficient approach compared to real-time API data extraction due to the aforementioned rate limits.


