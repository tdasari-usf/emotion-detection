import torch 
import streamlit as st
import pandas as pd
import numpy as np
from train import tokenizer, model
import torch.nn.functional as F

option = st.sidebar.selectbox("Projects", ("Emotion detections", 
												"Project 2"))

st.header(option)

if option == "Emotion detections":
	st.write("A classifier to label annoyance or admiration content")
	text = st.text_input("Enter text to be classified")
	print(text)
	model.eval()
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	encoded_texts = tokenizer([text], padding=True, truncation=True, return_tensors='pt').to(device)
	outputs = model(encoded_texts['input_ids'], attention_mask=encoded_texts['attention_mask'])
	outs = F.softmax(outputs.logits, dim = 1)
	predicted_labels = torch.argmax(outs, dim = 1)
	#assign label 1 to admiration, 0 to annoyance
	label_dict = { 0:'annoyance', 1: 'admiration'}
	st.write(text, predicted_labels.item())
	st.write(outs)
	st.write(label_dict[predicted_labels.item()])
	
if option == "Project 2":
	st.write('In progress...')
	