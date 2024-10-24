# NLP-TEXT-MINING-PROJECT

**Project: 360DigitMG Reviews**

<div align = "justify">
This project performs text mining and sentiment analysis on customer reviews of 360DigitMG an IT institute, obtained from the SiteJabber website. The primary tasks include scraping the reviews, cleaning the data, performing sentiment analysis, and visualizing the results through word clouds and bigram frequency analysis.
</div>
-

<div align = "justify">

**Project Steps:**

- **Data Collection**: Reviews were scraped from SiteJabber's 360DigitMG page using BeautifulSoup. The scraped data was stored in a pandas DataFrame for further processing.

- **Data Cleaning**: The reviews were preprocessed by converting text to lowercase, removing punctuation, and eliminating stopwords using the nltk library. Spell correction was applied to enhance data quality.

- **Sentiment Analysis**: Sentiment analysis was performed using the VADER sentiment analyzer to determine the polarity (positive or negative) of each review. The sentiment score was added to the DataFrame, and the results were visualized using a bar chart.

- **Word Cloud Generation**: A general word cloud was created to visualize the most frequently occurring words in the reviews as shown below.

![image](https://github.com/user-attachments/assets/e45ef176-5688-4db8-ab45-733c914f5d70)


Additionally, separate word clouds for positive and negative words were generated by filtering the reviews using predefined lists of positive and negative words as below.

![image](https://github.com/user-attachments/assets/e6acd511-66e6-470c-b450-114215499511)

![image](https://github.com/user-attachments/assets/272f6795-9e58-4377-85a1-d75ceb6cb426)


- **Bigram Frequency Analysis:**
Bigram analysis was performed to identify common two-word combinations in the reviews. A word cloud was generated to visualize the most frequent bigrams as displayed below.

![image](https://github.com/user-attachments/assets/6200959b-66e1-4430-b1be-4a7a53303d2e)
</div>

**Libraries Used:** 
- BeautifulSoup and requests for web scraping
- pandas and numpy for data manipulation
- nltk and TextBlob for text preprocessing
- VADER for sentiment analysis
- WordCloud and matplotlib for visualization
- CountVectorizer from sklearn for bigram analysis

<div align = "justify">This project provides insights into customer feedback by identifying common themes and sentiments in the reviews. The visualizations, especially the word clouds, offer an intuitive representation of frequently used terms and customer sentiments.</div>
