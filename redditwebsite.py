from flask import Flask, render_template, request
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
import torch
import os

script_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_path, "model_2bins")
print(model_path)

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)


app = Flask(__name__)

@app.route("/", methods = ["GET", "POST"])

def hello_world():
	user_input = None
	if request.method == "POST":
		user_input = request.form.get("my_textbox")
		enc = tokenizer([user_input], padding=True,truncation=True,return_tensors="pt")

		with torch.no_grad():
			outputs = model(**enc)
			logits = outputs.logits
			preds = torch.argmax(logits, dim=1)

		label_years = ["2008-2010", "2020-2022"]

		return render_template("guessed.html", user_input=user_input,prediction=label_years[preds.tolist()[0]])
	else:
		return render_template("main.html")