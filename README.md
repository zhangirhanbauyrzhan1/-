
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from datasets import load_dataset
import re

print("Загружаем датасет Twitter US Airline Sentiment...")
dataset = load_dataset("tweet_eval", "sentiment")
df = pd.DataFrame(dataset['train'])

df = df[['text', 'label']]
df['label'] = df['label'].map({0: 'negative', 1: 'neutral', 2: 'positive'})

print("Выполняем разметку на основе правил...")


def rule_based_labeling(text):
    positive_keywords = ['good', 'excellent', 'love', 'amazing', 'awesome', 'great']
    negative_keywords = ['bad', 'worst', 'hate', 'terrible', 'awful', 'poor']

    if any(word in text.lower() for word in positive_keywords):
        return 'positive'
    elif any(word in text.lower() for word in negative_keywords):
        return 'negative'
    else:
        return 'neutral'


df['rule_label'] = df['text'].apply(rule_based_labeling)

print("Симулируем ручную разметку...")
manual_subset = df.sample(100, random_state=42)
manual_subset['manual_label'] = manual_subset['text'].apply(lambda x: 'positive' if 'love' in x else 'negative')

print("Объединяем данные...")
combined_df = df.copy()
combined_df.update(manual_subset[['manual_label']])
combined_df['final_label'] = combined_df['manual_label'].combine_first(combined_df['rule_label'])

print("Обучаем модель...")
train_df, test_df = train_test_split(combined_df, test_size=0.2, random_state=42)
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_df['text'])
y_train = train_df['final_label']
X_test = vectorizer.transform(test_df['text'])
y_test = test_df['final_label']

model = MultinomialNB()
model.fit(X_train, y_train)

print("Оцениваем модель...")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

combined_df.to_csv("labeled_data.csv", index=False)
print("Данные сохранены в 'labeled_data.csv'.")
