## Updates/Log
[Updates/Log](#updateslog)
   - [10/02/2023](#date-10022023)
   - [10/20/2023](#date-10202023)
   - [MM/DD/YYYY](#date-mmddyyyy)
   - [Future Updates](#future-updates)

### [Date: 10/20/2023]
#### Progress
- Successfully implemented the initial version of the RNN model for sentiment analysis, achieving an accuracy of 80% for test data.
- Collected additional data to enhance the training set.

#### Challenges
- The model's performance on unseen data, specifically related to stock posts, is suboptimal with an accuracy of around 60%.
- Recognized that leveraging existing datasets might be a more efficient strategy than extracting data in real-time from APIs, especially considering the rate limits associated with the latter.

#### Next Steps
- Plan to develop a mechanism to pinpoint the specific stock that a post refers to.
- Will need to implement a method to identify the stock a post is talking about.

#### To-Dos
- Conduct a thorough EDA on the extracted stock data, focusing on identifying potential correlations with expected social media sentiment trends.
- Identify and acquire datasets containing social media posts and comments related to the meme stocks.
- Develop a strategy for cleaning and preprocessing the social media data, ensuring it is in a usable format for sentiment analysis.
- Develop a strategy to predict stock movement using stock data.
- Add market sentiment to the model predicting stock movement.

### [Date: 10/02/2023]
#### Progress
- Successfully extracted stock data from Yahoo Finance, focusing on popular meme stocks like GME, AMC, and BB.
- Initiated exploratory data analysis (EDA) to identify general trends and patterns within the extracted stock data, such as price fluctuations and trading volumes during specific time frames.

#### Challenges
- Encountered API rate limits while extracting data, which hindered the efficiency of the data retrieval process.
- Realized that searching and utilizing available datasets might be a more efficient approach compared to real-time API data extraction due to the aforementioned rate limits.


