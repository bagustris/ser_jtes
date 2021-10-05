import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import AutoConfig, Wav2Vec2FeatureExtractor
from src.models import Wav2Vec2ForSpeechClassification

import librosa
# import IPython.display as ipd
import numpy as np
import pandas as pd

from datasets import load_dataset, load_metric
from sklearn.metrics import classification_report


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model_name_or_path = "m3hrdadfi/wav2vec2-xlsr-greek-speech-emotion-recognition"
model_name_or_path = '/home/aad13432ni/github/ser-jtes-wav2vec2/models/checkpoint-21600/'
config = AutoConfig.from_pretrained(model_name_or_path)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)
sampling_rate = feature_extractor.sampling_rate
model = Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path).to(device)


def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    speech_array = speech_array.squeeze().numpy()
    speech_array = librosa.resample(np.asarray(speech_array), sampling_rate, feature_extractor.sampling_rate)

    batch["speech"] = speech_array
    return batch


def predict(batch):
    features = feature_extractor(batch["speech"], sampling_rate=feature_extractor.sampling_rate, return_tensors="pt", padding=True)

    input_values = features.input_values.to(device)

    with torch.no_grad():
        logits = model(input_values).logits 

    pred_ids = torch.argmax(logits, dim=-1).detach().cpu().numpy()
    batch["predicted"] = pred_ids
    return batch
    
# batch prediction
test_dataset = load_dataset("csv", 
    data_files={"test": "content/data/test.csv"}, 
                delimiter="\t")["test"]


test_dataset = test_dataset.map(speech_file_to_array_fn)
result = test_dataset.map(predict, batched=True, batch_size=2)
label_names = [config.id2label[i] for i in range(config.num_labels)]
print(label_names)

y_true = [config.label2id[name] for name in result["emotion"]]
y_pred = result["predicted"]
print(classification_report(y_true, y_pred, target_names=label_names))