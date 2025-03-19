import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
import json
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading required NLTK data...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger')
    print("NLTK data download complete!")

class TextPreprocessor:
    def __init__(self, min_freq=1, max_vocab_size=10000):
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vocab = None
        
    def create_vocab(self, text_tokens):
        word_counts = Counter(text_tokens)
        self.vocab = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<START>': 2,
            '<END>': 3
        }
        
        # For small test cases, include all words
        if len(word_counts) < 100:  # If we have less than 100 unique words
            for word in word_counts:
                if word not in self.vocab:
                    self.vocab[word] = len(self.vocab)
        else:
            # For larger datasets, use frequency threshold
            for word, count in word_counts.most_common(self.max_vocab_size - len(self.vocab)):
                if count >= self.min_freq:
                    self.vocab[word] = len(self.vocab)
                
    def preprocess_text(self, text):
        text = text.lower()    
        # Handle contractions
        contractions = {
            "n't": " not",
            "'ll": " will",
            "'re": " are",
            "'ve": " have",
            "'m": " am",
            "'d": " would"
        }
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # Remove URLs and emails
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove numbers but keep currency symbols
        text = re.sub(r'\b\d+\b', '', text)
        
        # Remove special characters but keep apostrophes
        text = re.sub(r'[^\w\s\']', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Process tokens
        processed_tokens = []
        for token in tokens:                
            # Handle special cases
            if token.startswith("'") or token.endswith("'"):
                token = token.strip("'")
                
            # Skip if token is empty after cleaning
            if not token:
                continue
                
            # Lemmatize the token
            lemmatized = self.lemmatizer.lemmatize(token)
            
            # Only add if the lemmatized token is not empty
            if lemmatized:
                processed_tokens.append(lemmatized)
                
        return processed_tokens
    
    def create_sequences(self, processed_tokens, seq_length):
        sequences = []
        targets = []
        
        for i in range(len(processed_tokens) - seq_length):
            sequence = processed_tokens[i:i + seq_length]
            # Create targets for each position in the sequence
            sequence_targets = processed_tokens[i+1:i + seq_length + 1]
            
            # Convert to indices
            sequence = [self.vocab.get(word, self.vocab['<UNK>']) for word in sequence]
            sequence_targets = [self.vocab.get(word, self.vocab['<UNK>']) for word in sequence_targets]
            
            sequences.append(sequence)
            targets.append(sequence_targets)
            
        return sequences, targets

class TextDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.tensor(sequences)
        self.targets = torch.tensor(targets)
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class LightweightTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=2, dim_feedforward=256, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 16, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length)
        x = self.embedding(x)  # (batch_size, seq_length, d_model)
        x = x + self.pos_encoder[:, :x.size(1), :]  # Add positional encoding
        x = self.transformer_encoder(x)
        x = self.decoder(x)
        return x

def train_model(model, train_loader, optimizer, criterion, device, vocab, num_epochs=5):
    model.train()
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)  # Shape: (batch_size, seq_length, vocab_size)
            
            # Calculate loss without reshaping
            loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')
        
        # Save model after each epoch
        model_path = f'saved_models/model_epoch{epoch+1}_loss{avg_loss:.4f}.pt'
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'vocab': vocab  # Save vocabulary
        }, model_path)
        print(f'Model saved to {model_path}')
        
        # Update best loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save best model
            best_model_path = f'saved_models/best_model_loss{int(best_loss*10)}.pt'
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'vocab': vocab  # Save vocabulary
            }, best_model_path)
            print(f'Best model saved to {best_model_path}')

def predict_next_word(model, sequence, vocab, device, top_k=5):
    model.eval()
    with torch.no_grad():
        sequence = [vocab.get(word, vocab['<UNK>']) for word in sequence]
        sequence = torch.tensor(sequence).unsqueeze(0).to(device)
        output = model(sequence)
        probs = F.softmax(output[0, -1], dim=0)
        top_probs, top_indices = torch.topk(probs, top_k)
        
        # Convert indices to words
        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            word = [k for k, v in vocab.items() if v == idx.item()][0]
            predictions.append((word, prob.item()))
            
        return predictions

def evaluate_all_models(test_sequences, device):
    print("\n=== Evaluating All Saved Models ===")
    print("-" * 50)
    
    # Get all model files from saved_models directory
    model_files = [f for f in os.listdir('saved_models') if f.endswith('.pt')]
    model_files.sort()  # Sort to evaluate in chronological order
    
    for model_file in model_files:
        print(f"\nEvaluating model: {model_file}")
        print("-" * 30)
        
        # Load model and vocabulary
        checkpoint = torch.load(os.path.join('saved_models', model_file))
        vocab = checkpoint['vocab']
        model = LightweightTransformer(len(vocab)).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Print model info
        print(f"Epoch: {checkpoint['epoch']}")
        print(f"Loss: {checkpoint['loss']:.4f}")
        print(f"Vocabulary size: {len(vocab)}")
        
        # Make predictions
        for sequence in test_sequences:
            predictions = predict_next_word(model, sequence, vocab, device, top_k=3)
            print(f"\nInput: {' '.join(sequence)}")
            print("Top 3 predictions:")
            for i, (word, prob) in enumerate(predictions, 1):
                print(f"{i}. {word} ({prob:.2%})")
        print("-" * 30)

def main():
    with open('dataset.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    preprocessor = TextPreprocessor(min_freq=1, max_vocab_size=10000)
    processed_tokens = preprocessor.preprocess_text(text)
    preprocessor.create_vocab(processed_tokens)
    
    seq_length = 16
    batch_size = 128
    
    sequences, targets = preprocessor.create_sequences(processed_tokens, seq_length)
    dataset = TextDataset(sequences, targets)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LightweightTransformer(len(preprocessor.vocab)).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Print some information about the dataset
    print(f"\nDataset Information:")
    print(f"Vocabulary size: {len(preprocessor.vocab)}")
    print(f"Number of sequences: {len(sequences)}")
    print(f"Sequence length: {seq_length}")
    print(f"Batch size: {batch_size}")
    
    train_model(model, train_loader, optimizer, criterion, device, preprocessor.vocab)
    
    test_sequences = [
        ["this", "is", "a", "test", "sentence", "I", "want", "to", "see"],
        ["the", "weather", "is", "very", "nice" "today", "I", "wonder"],
        ["i", "would", "like", "to", "go"]
    ]
    
    # Evaluate all saved models
    evaluate_all_models(test_sequences, device)

def test_preprocessor():
    preprocessor = TextPreprocessor(min_freq=2, max_vocab_size=30000)
    text = "This is a test sentence. The weather is very nice. I would like to go."
    
    # Process text and show results
    processed_tokens = preprocessor.preprocess_text(text)
    print("\nProcessed tokens:", processed_tokens)
    print("Number of tokens:", len(processed_tokens))
    
    # Create vocabulary
    preprocessor.create_vocab(processed_tokens)
    print("\nVocabulary size:", len(preprocessor.vocab))
    
    # Show vocabulary in a more readable format
    print("\nVocabulary:")
    for word, idx in sorted(preprocessor.vocab.items(), key=lambda x: x[1]):
        print(f"{word}: {idx}")
    
    # Create sequences with shorter sequence length
    seq_length = 3  # Reduced from 10 to 3
    sequences, targets = preprocessor.create_sequences(processed_tokens, seq_length)
    print("\nNumber of sequences:", len(sequences))
    
    # Create and test dataset
    dataset = TextDataset(sequences, targets)
    print("\nDataset size:", len(dataset))
    
    if len(dataset) > 0:
        print("\nFirst sequence and target:")
        sequence, target = dataset[0]
        print("Sequence indices:", sequence.tolist())
        print("Target index:", target.item())
        print("\nSequence words:", [k for k, v in preprocessor.vocab.items() if v in sequence.tolist()])
        print("Target word:", [k for k, v in preprocessor.vocab.items() if v == target.item()][0])
    else:
        print("\nNo sequences created! Try reducing sequence_length or increasing input text.")

def test_dataset_creation():
    print("\n=== Testing Dataset Creation ===")
    
    # Test with a simple text
    text = "This is a test sentence. The weather is very nice. I would like to go."
    preprocessor = TextPreprocessor(min_freq=2, max_vocab_size=10000)
    
    # Process text
    processed_tokens = preprocessor.preprocess_text(text)
    print("\n1. Text Processing:")
    print("Original text:", text)
    print("Processed tokens:", processed_tokens)
    print("Number of tokens:", len(processed_tokens))
    
    # Create vocabulary
    preprocessor.create_vocab(processed_tokens)
    print("\n2. Vocabulary Creation:")
    print("Vocabulary size:", len(preprocessor.vocab))
    print("First 5 vocabulary items:")
    for word, idx in list(preprocessor.vocab.items())[:5]:
        print(f"{word}: {idx}")
    
    # Create sequences
    seq_length = 3
    sequences, targets = preprocessor.create_sequences(processed_tokens, seq_length)
    print("\n3. Sequence Creation:")
    print("Number of sequences:", len(sequences))
    print("Number of targets:", len(targets))
    
    # Create dataset
    dataset = TextDataset(sequences, targets)
    print("\n4. Dataset Creation:")
    print("Dataset size:", len(dataset))
    
    # Test first sequence
    if len(dataset) > 0:
        sequence, target = dataset[0]
        print("\n5. First Sequence Details:")
        print("Sequence shape:", sequence.shape)
        print("Target shape:", target.shape)
        print("Sequence indices:", sequence.tolist())
        print("Target index:", target.item())
        print("\nSequence words:", [k for k, v in preprocessor.vocab.items() if v in sequence.tolist()])
        print("Target word:", [k for k, v in preprocessor.vocab.items() if v == target.item()][0])
        
        # Test batch creation
        batch_size = 2
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        batch = next(iter(dataloader))
        print("\n6. Batch Creation:")
        print("Batch sequence shape:", batch[0].shape)
        print("Batch target shape:", batch[1].shape)
        print("Expected shapes:")
        print(f"- Sequences: torch.Size([{batch_size}, {seq_length}])")
        print(f"- Targets: torch.Size([{batch_size}, {seq_length}])")

def test_model_tensor_dimensions():
    print("\n=== Testing Model Tensor Dimensions ===")
    
    # Create a small test vocabulary
    vocab_size = 100
    preprocessor = TextPreprocessor(min_freq=2, max_vocab_size=10000)
    preprocessor.vocab = {f"word_{i}": i for i in range(vocab_size)}
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LightweightTransformer(vocab_size).to(device)
    
    # Test with different sequence lengths
    test_sequences = [3, 5, 10]
    batch_size = 2
    
    print("\nModel Architecture:")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Embedding dimension: {model.embedding.embedding_dim}")
    print(f"Number of transformer layers: {len(model.transformer_encoder.layers)}")
    
    for seq_length in test_sequences:
        print(f"\nTesting sequence length: {seq_length}")
        
        # Create dummy input
        dummy_input = torch.randint(0, vocab_size, (batch_size, seq_length)).to(device)
        print(f"Input shape: {dummy_input.shape}")
        
        # Forward pass
        try:
            output = model(dummy_input)
            print(f"Output shape: {output.shape}")
            print("✓ Tensor dimensions match expected shapes")
        except Exception as e:
            print(f"✗ Error in forward pass: {str(e)}")
        
        # Verify shapes
        expected_output_shape = (batch_size, seq_length, vocab_size)
        if output.shape == expected_output_shape:
            print("✓ Output shape matches expected shape")
        else:
            print(f"✗ Output shape mismatch. Expected {expected_output_shape}, got {output.shape}")

if __name__ == "__main__":
    main()