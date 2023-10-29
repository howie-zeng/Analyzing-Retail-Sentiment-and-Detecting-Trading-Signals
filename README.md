## Updates/Log
- [10/02/2023](#date-10022023)
- [10/20/2023](#date-10202023)
- [10/24/2023](#date-10242023)
- [Future Updates](#future-updates)

### Date: 10/24/2023
#### Third to Fourth Week Progress
- Transitioned from the XGBoost Classifier to the Regressor, achieving a 10x speed enhancement but with a minor 10% performance reduction.
- Introduced new evaluation metrics; residuals now appear random.
- Enriched the XGBoost model with numerous indicators, resulting in a two-thirds reduction in the mean absolute error percentage.
- Improved model's capability in detecting selling signals.
- Recorded an absolute mean error percentage of 1.87% over a decade and 1.74% over two years.
- Incorporated advanced indicators like the Williams VIX Fix and the RSI Bollinger Strategy.
- Conducted experiments on sectors such as SPY, QQQ, and DIA, as well as with stocks like TSLA and Amazon.
- Implemented a Volume Indicator for tracking volume outbreaks.

#### Upcoming Objectives
- Prioritize enhanced feature selection.
- Post enhancements, explore alternative models like LSM.
- Address the colinearity issue.

### Date: 10/20/2023
#### Second Week Challenges
- The model displayed mediocre performance on unseen data related to stock posts, achieving only around 60% accuracy.
- Identified the potential advantage of using pre-existing datasets over real-time API data extraction, mainly due to rate limit restrictions of APIs.
- Need to delve deeper into prediction models to optimize performance.
- The current stock name extractor overlooks detailed sentiment analysis, focusing solely on the stock name mentioned in comments.

#### Upcoming Objectives
- Initiate the development of a mechanism to accurately identify the specific stock referenced in a post.
- Further enhancement is required for the method determining which stock a post is referring to.

#### Tasks in Queue
- Perform an in-depth EDA on the stock data, emphasizing the discovery of correlations with anticipated social media sentiment patterns.
- Search and procure datasets containing social media posts and comments pertinent to meme stocks.
- Formulate a robust methodology for cleaning and preprocessing the social media data, ensuring its viability for sentiment analysis.
- Design a strategy for predicting stock movements using the accumulated stock data.
- Integrate market sentiment into the stock movement prediction model.

### Date: 10/02/2023
#### First Week Progress
- Successfully harvested stock data from Yahoo Finance, with a spotlight on renowned meme stocks like GME, AMC, and BB.
- Embarked on an exploratory data analysis (EDA) journey to decipher overarching trends and patterns in the stock data, such as discernible price swings and distinct trading volume periods.

#### Encountered Issues
- Faced constraints with API rate limits during data extraction, impeding the speed and efficiency of data acquisition.
- Acknowledged the potential efficiency of sourcing and employing available datasets over real-time API data extraction, primarily due to the constraints mentioned above.
