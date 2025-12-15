# RETINAL EYE DETECTION

A full-stack AI-powered web application for detecting retinal eye diseases using deep learning models.
Users can upload a retinal fundus image, and the system predicts whether it belongs to CNV, DME, DRUSEN, or NORMAL categories.

## 📌 Project Overview
Retinal diseases are a major cause of vision impairment. Early detection can significantly reduce the risk of permanent vision loss.
This project leverages deep learning and computer vision to assist in the automated detection of retinal eye diseases through a simple and user-friendly web interface.

## 🚀 Features
- 🖼️ Upload retinal eye images via a web interface

- 🤖 Multiple deep learning models used:
  
		-MobileNet
		-ResNet
		-EfficientNet
		-Custom CNN
- 🧠 Model comparison to select the best prediction

- 📊 Displays:

  		-Predicted disease class

  		-Confidence percentage

		-Best performing model

## 🧠 Technology Stack
### Frontend

-React (Vite)

-HTML5, CSS3, JavaScript

-Fetch API for backend communication

### Backend

-Python

-Flask

-TensorFlow

-NumPy


## 📂 Project Structure
retinal-eye-detection/
│
├── backend/
│   ├── app.py               # Flask backend API
│   ├── requirement.txt      # Python dependencies
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main React component
│   │   ├── App.css
│   │   ├── main.jsx
│   │   └── assets/
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
│
├── public/
│
├── .gitignore
└── README.md


















