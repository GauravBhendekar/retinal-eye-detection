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

- React (Vite)
- HTML5, CSS3, JavaScript
- Fetch API for backend communication

### Backend

- Python

- Flask

- TensorFlow

- NumPy


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

##⚙️ How to Run the Project Locally
1️. Clone the Repository

			- git clone https://github.com/GauravBhendekar/retinal-eye-detection.git			
			- cd retinal-eye-detection

2. Download Trained ML Models

⚠️ Important Note

The trained deep learning model files are not included in this repository due to GitHub file size limitations.

📥 Download models from Google Drive:
	
	🔗 https://drive.google.com/drive/folders/17zDbCcgT--7K1-GNqrE84TQnPY-F-0iS?usp=sharing

After downloading, place all model files inside:

	backend/

3. Run Backend (Flask Server)
	
		- cd backend
		- pip install -r requirement.txt
		- python app.py

Backend will start at:

	- http://127.0.0.1:5000

4. Run Frontend (React Application)

	   - http://localhost:5173

## 🧪 Disease Classes

- CNV – Choroidal Neovascularization

- DME – Diabetic Macular Edema

- DRUSEN – Accumulation of extracellular material

- NORMAL – Healthy retina

⚠️ Notes & Limitations

- This application is intended for educational and research purposes only

- Not approved for clinical or medical diagnosis

- Model accuracy depends on image quality and training dataset

- Trained model files are excluded using .gitignore
   

















