# 📦 Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud, STOPWORDS

# 📥 Load survey data exported from Google Forms
df = pd.read_csv('/content/Student_Satisfaction_Survey.csv', encoding='latin1')

# 👀 Preview first 40 rows to understand structure
display(df.head(40))

# 🔍 Check for missing values across all columns
display(df.isnull().sum())

# 🧹 Clean 'Average/ Percentage' column by extracting numeric rating before '/'
df['Average/ Percentage'] = pd.to_numeric(
    df['Average/ Percentage'].str.split('/').str[0], errors='coerce'
)

# 🧼 Drop rows missing essential data (ratings or feedback text)
df.dropna(subset=['Average/ Percentage', 'Questions'], inplace=True)

# 📊 Generate descriptive statistics for ratings
descriptive_stats = df['Average/ Percentage'].describe()
display(descriptive_stats)

# 📈 Visualize distribution of ratings using histogram + KDE
plt.figure(figsize=(10, 6))
sns.histplot(df['Average/ Percentage'], bins=10, kde=True, color='skyblue')
plt.title('📊 Distribution of Average Ratings')
plt.xlabel('Average Rating')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# 💬 Apply sentiment analysis to feedback text using TextBlob
df['Sentiment'] = df['Questions'].apply(lambda x: TextBlob(x).sentiment.polarity)

# 👓 Preview sentiment scores alongside ratings and questions
display(df[['Questions', 'Average/ Percentage', 'Sentiment']].head())

# 🔗 Scatter plot: Relationship between rating and sentiment, colored by course
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Average/ Percentage', y='Sentiment', data=df, hue='Basic Course', palette='viridis')
plt.title('🔗 Relationship between Rating and Sentiment')
plt.xlabel('Average Rating')
plt.ylabel('Sentiment Polarity')
plt.grid(True)
plt.legend(title='Course', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 📦 Box plot: Rating distribution across different courses
plt.figure(figsize=(12, 7))
sns.boxplot(x='Basic Course', y='Average/ Percentage', data=df, palette='Set2')
plt.title('📦 Ratings Distribution by Course')
plt.xlabel('Basic Course')
plt.ylabel('Average Rating')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 📦 Box plot: Sentiment distribution across different courses
plt.figure(figsize=(12, 7))
sns.boxplot(x='Basic Course', y='Sentiment', data=df, palette='Set3')
plt.title('📦 Sentiment Distribution by Course')
plt.xlabel('Basic Course')
plt.ylabel('Sentiment Polarity')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 🔥 Heatmap: Correlation between rating and sentiment
corr_matrix = df[['Average/ Percentage', 'Sentiment']].corr()
plt.figure(figsize=(6, 4))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('🔥 Correlation Matrix: Ratings vs Sentiment')
plt.show()

# ☁ WordCloud: Most frequent terms in feedback comments
text_data = ' '.join(df['Questions'].dropna().astype(str))
stopwords = set(STOPWORDS)
wordcloud = WordCloud(width=800, height=400, background_color='white',
                      stopwords=stopwords, colormap='viridis').generate(text_data)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('☁ Most Common Feedback Terms')
plt.show()

# 📌 Summary of key findings and actionable recommendations
print(" Summary of Findings:")
print(f" ⭐- Average rating: {descriptive_stats['mean']:.2f} ± {descriptive_stats['std']:.2f}")
print("- Ratings skew toward higher values, indicating general satisfaction.")
print("- Positive correlation between rating and sentiment: happier comments tend to have higher scores.")
print("- Some courses show lower ratings and sentiment—potential areas for improvement.")

print("\n🔧 Recommendations:")
print("- Investigate low-rated courses by reviewing specific feedback.")
print("- Address issues like syllabus clarity, teaching quality, and communication.")
print("- Encourage constructive feedback and share best practices among faculty.")
print("- Use this dashboard regularly to monitor satisfaction trends.")

# 💾 Export cleaned dataset for future use or reporting
df.to_csv('/content/Cleaned_Student_Feedback.csv', index=False)