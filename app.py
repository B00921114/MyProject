# Import necessary libraries
import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from difflib import SequenceMatcher

# Download NLTK stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# Preprocess text function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Google Scholar search function using SerpAPI
def search_google_scholar(query, api_key, num_results=20):
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google_scholar",
        "q": query,
        "num": num_results,
        "api_key": api_key
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return f"Error: {response.status_code} - {response.text}"

# Prepare text data from search results
def prepare_text_data(results):
    texts = []
    for result in results.get("organic_results", []):
        title = result.get("title", "")
        snippet = result.get("snippet", "")
        full_text = title + " " + snippet
        cleaned_text = preprocess_text(full_text)
        texts.append(cleaned_text)
    return texts

# Collect Data Using the API
api_key = "e73437c2007190db079d36557402977fc4e68a641bcc4fb9f5e43df14f18c950"  # Replace with your SerpAPI key
queries = ["machine learning in healthcare", "artificial intelligence in medicine", "deep learning in medical imaging"]
texts = []

# Collect data using the API
for query in queries:
    results = search_google_scholar(query, api_key, num_results=20)
    if isinstance(results, dict):
        texts += prepare_text_data(results)

# Tokenize the Data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

input_ids = []
attention_masks = []
for text in texts:
    encoded_dict = tokenizer(
        text,
        add_special_tokens=True,
        max_length=64,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

# Convert lists to tensors
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor([0] * len(input_ids))  # Dummy labels for unsupervised learning

# Create a DataLoader
from torch.utils.data import DataLoader, TensorDataset

train_dataset = TensorDataset(input_ids, attention_masks, labels)
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Fine-Tune BERT with Reduced Batch Size and Epochs
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)  # Dummy task with 2 labels

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,  # Reduced from 3 to 1
    per_device_train_batch_size=4,  # Reduced from 8 to 4
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Training loop
model.train()
for epoch in range(training_args.num_train_epochs):
    for batch in train_dataloader:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch  # Unpack the batch
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Use the Fine-Tuned Model for Recommendations
def get_bert_embeddings(texts, model, tokenizer):
    model.eval()  # Set model to evaluation mode
    embeddings = []
    with torch.no_grad():  # Disable gradient computation for efficiency
        for text in texts:
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
            outputs = model.bert(**inputs)  # Access the BERT part of the model
            embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
            embeddings.append(embedding)
    return np.vstack(embeddings)

# Example Usage
query = "machine learning in healthcare"
user_profile_embedding = get_bert_embeddings([query], model, tokenizer)

recommendations = search_google_scholar(query, api_key, num_results=20)
texts = prepare_text_data(recommendations)
paper_embeddings = get_bert_embeddings(texts, model, tokenizer)
similarities = cosine_similarity(paper_embeddings, user_profile_embedding.reshape(1, -1))
ranked_indices = np.argsort(similarities[:, 0])[::-1]
recommended_papers = [(recommendations['organic_results'][i]['title'], recommendations['organic_results'][i]['link'], similarities[i][0]) for i in ranked_indices[:5]]

# Display the recommendations
for title, link, score in recommended_papers:
    print(f"Title: {title}")
    print(f"Link: {link}")
    print(f"Similarity Score: {score:.4f}")
    print("-" * 80)

# Precision@K, Recall@K, F1 Score, and MAP Calculation
def normalize_title(title):
    title = title.lower()
    title = re.sub(r'[^a-zA-Z0-9\s]', '', title)
    return title.strip()

def partial_match_score(a, b):
    return SequenceMatcher(None, a, b).ratio()

def precision_at_k_partial(actual, predicted, k, threshold=0.8):
    predicted_at_k = predicted[:k]
    hits = 0
    for pred in predicted_at_k:
        if any(partial_match_score(pred, act) > threshold for act in actual):
            hits += 1
    precision = hits / len(predicted_at_k)
    return precision

def recall_at_k(actual, predicted, k, threshold=0.8):
    predicted_at_k = predicted[:k]
    hits = 0
    for act in actual:
        if any(partial_match_score(act, pred) > threshold for pred in predicted_at_k):
            hits += 1
    recall = hits / len(actual)
    return recall

def f1_score(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

def average_precision_at_k(actual, predicted, k):
    if not actual:
        return 0.0
    
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:  # only count hits once
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(actual), k)

def mean_average_precision_at_k(actual_list, predicted_list, k=10):
    return np.mean([average_precision_at_k(a, p, k) for a, p in zip(actual_list, predicted_list)])

# Example ground truth for demonstration (this should be replaced with actual relevant papers for your use case)
actual_relevant_papers = [
    ["machine learning in healthcare a review", 
     "applications of machine learning in healthcare"]
]

predicted_papers = [normalize_title(title) for title, link, score in recommended_papers]

# Normalize ground truth titles
actual_relevant_papers = [[normalize_title(title) for title in actual_list] for actual_list in actual_relevant_papers]

# Calculate Precision, Recall, F1 Score, and MAP
k = 10
precision = precision_at_k_partial(actual_relevant_papers[0], predicted_papers, k)
recall = recall_at_k(actual_relevant_papers[0], predicted_papers, k)
f1 = f1_score(precision, recall)
map_score = mean_average_precision_at_k(actual_relevant_papers, [predicted_papers], k)

# Display results
print(f"Precision@{k}: {precision:.4f}")
print(f"Recall@{k}: {recall:.4f}")
print(f"F1 Score@{k}: {f1:.4f}")
print(f"MAP@{k}: {map_score:.4f}")

# Allow Users to Provide Feedback
def get_feedback(recommended_papers):
    feedback = []
    print("\nPlease rate the relevance of each recommended paper on a scale from 1 to 5 (1 = Not relevant, 5 = Very relevant):\n")
    for i, (title, link, score) in enumerate(recommended_papers, start=1):
        rating = int(input(f"Relevance of paper {i} (Title: {title}): "))
        feedback.append((title, link, score, rating))
    return feedback

# Store or Use Feedback for Further Analysis or Model Improvement
def analyze_feedback(feedback):
    avg_rating = np.mean([rating for _, _, _, rating in feedback])
    print(f"\nAverage feedback rating: {avg_rating:.2f}")
    if avg_rating < 3:
        print("Feedback indicates that the recommendations may need improvement.")
    else:
        print("Feedback indicates that the recommendations are generally relevant.")

# Placeholder function - replace with your actual logic
def get_user_query_and_recommend():
    """
    This function should handle:
    1. Getting the user's query
    2. Generating recommendations based on the query
    3. Returning the list of recommended papers
    """
    # Replace this with your actual implementation
    recommended_papers = [
        ("Sample Paper 1", "https://example.com/paper1", 0.9),
        ("Sample Paper 2", "https://example.com/paper2", 0.8),
        # ... add more recommendations here
    ]
    return recommended_papers

# Main loop for user interaction
def main():
    continue_interaction = True
    while continue_interaction:
        recommended_papers = get_user_query_and_recommend()
        feedback = get_feedback(recommended_papers)
        analyze_feedback(feedback)
        
        another_query = input("\nWould you like to enter another query? (yes/no): ").strip().lower()
        if another_query != 'yes':
            continue_interaction = False

# Run the interaction loop
if __name__ == "__main__":
    main()
