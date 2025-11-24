import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch

# Data preparation
sentiment_df = review[['text', 'stars']].dropna()

# Binary tags: 1 = positive (stars>=4), 0 = negative (stars<=2), neutral rating 3 Delete
sentiment_df = sentiment_df[sentiment_df['stars'] != 3]
sentiment_df['label'] = sentiment_df['stars'].apply(lambda x: 1 if x >= 4 else 0)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

sentiment_df['text'] = sentiment_df['text'].apply(clean_text)

train_texts, test_texts, train_labels, test_labels = train_test_split(
    sentiment_df['text'].tolist(),
    sentiment_df['label'].tolist(),
    test_size=0.2,
    random_state=42
)

# Tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)

class YelpDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = YelpDataset(train_encodings, train_labels)
test_dataset = YelpDataset(test_encodings, test_labels)

# Model & Trainer
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs'
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer
)

trainer.train()

eval_results = trainer.evaluate()
print("Evaluation results:", eval_results)

# Test set prediction & evaluation
preds_output = trainer.predict(test_dataset)
preds = preds_output.predictions.argmax(-1)

print("Accuracy:", accuracy_score(test_labels, preds))
print(classification_report(test_labels, preds))

# Export prediction results CSV
export_sentiment_df = pd.DataFrame({
    'text': test_texts,
    'actual_label': test_labels,
    'predicted_label': preds,
    'predicted_sentiment': ["Positive" if p==1 else "Negative" for p in preds]
})

export_sentiment_df.to_csv('sentiment_predictions.csv', index=False)
print("✅ Export completed: sentiment_predictions.csv")

# Example prediction function
def predict_sentiment_bert(new_reviews):
    encodings = tokenizer(new_reviews, truncation=True, padding=True, max_length=128, return_tensors='pt')
    outputs = model(**encodings)
    predictions = torch.argmax(outputs.logits, dim=-1)
    return ["Positive" if p==1 else "Negative" for p in predictions]

sample_reviews = [
    "The service was amazing and the food was delicious!",
    "Horrible experience, will never return."
]

print(predict_sentiment_bert(sample_reviews))
