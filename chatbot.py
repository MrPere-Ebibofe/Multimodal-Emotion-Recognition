import torch
import torch.nn as nn
import torchaudio
import sounddevice as sd
import numpy as np
import speech_recognition as sr
from transformers import BertModel, BertTokenizer

# Define Multimodal Emotion Recognition Model
class MultimodalEmotionModel(nn.Module):
    def __init__(self, num_labels):
        super(MultimodalEmotionModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.audio_fc = nn.Linear(100, 768)  # Project 100D audio features to 768D
        self.fc = nn.Linear(768 + 768, num_labels)  # Fusion layer

    def forward(self, input_ids=None, attention_mask=None, audio_features=None):
        if input_ids is not None:
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            text_features = bert_output.last_hidden_state[:, 0, :]  # CLS token representation
        else:
            text_features = torch.zeros(1, 768)  # If no text input, use a zero vector

        if audio_features is not None:
            audio_features = self.audio_fc(audio_features)  # Project audio to 768D
        else:
            audio_features = torch.zeros(1, 768)  # If no audio input, use a zero vector

        combined_features = torch.cat((text_features, audio_features), dim=1)
        return self.fc(combined_features)

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
num_labels = 7  # Modify based on your emotion classes
model = MultimodalEmotionModel(num_labels)
model.load_state_dict(torch.load("multimodal_emotion_model.pt", map_location=torch.device("cpu")))
model.eval()

# Function to record audio (5 seconds) and extract features
def record_audio(duration=5, samplerate=16000):
    print("🎙️ Recording audio... Speak now!")
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype="float32")
    sd.wait()
    return np.squeeze(audio_data)

# Function to extract MFCC features
def extract_audio_features(audio_data, sample_rate=16000):
    waveform = torch.tensor(audio_data).unsqueeze(0)  # Add batch dimension
    mfcc_transform = torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=100)
    mfcc_features = mfcc_transform(waveform)
    return mfcc_features.squeeze(0).mean(dim=1).unsqueeze(0)  # Reshape for model

# Function to process user input
def chatbot_interaction():
    print("\n💬 Type 'exit' to quit the chatbot.")

    while True:
        mode = input("\n🎙️ Do you want to use (1) Text, (2) Audio, or (3) Both? Enter 1, 2, or 3: ").strip()

        input_ids, attention_mask, audio_features = None, None, None

        if mode == "1":  # Text-only
            user_text = input("\n📝 You: ")
            if user_text.lower() == "exit":
                print("👋 Goodbye!")
                break
            inputs = tokenizer(user_text, return_tensors="pt", padding=True, truncation=True)
            input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]

        elif mode == "2":  # Audio-only
            audio_data = record_audio()
            audio_features = extract_audio_features(audio_data)

        elif mode == "3":  # Both Text & Audio
            user_text = input("\n📝 You: ")
            if user_text.lower() == "exit":
                print("👋 Goodbye!")
                break
            inputs = tokenizer(user_text, return_tensors="pt", padding=True, truncation=True)
            input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]

            audio_data = record_audio()
            audio_features = extract_audio_features(audio_data)

        else:
            print("⚠️ Invalid choice. Please enter 1, 2, or 3.")
            continue

        # Run through the model
        with torch.no_grad():
            outputs = model(input_ids, attention_mask, audio_features)

        # Get predicted emotion (softmax not applied)
        predicted_emotion = torch.argmax(outputs, dim=1).item()
        emotion_labels = ["Neutral", "Happy", "Sad", "Angry", "Surprised", "Disgust", "Fear"]

        print(f"🤖 Chatbot: Detected Emotion → {emotion_labels[predicted_emotion]}")
        print("💡 (Next step: Generate response based on detected emotion...)")

# Run chatbot
if __name__ == "__main__":
    chatbot_interaction()
