# 🤖 Machine Learn to Talk  
### Machine Status / Error Prediction using Machine Learning

## 📌 Overview  
Machine Learn to Talk is a Machine Learning-based web application that predicts machine status and identifies potential errors in real-time. The system uses a Random Forest Classifier trained on historical machine data to classify operational states such as production, technical issues, changeovers, and more. It helps reduce downtime, improve predictive maintenance, and optimize manufacturing efficiency.

## 🚀 Features  
- Real-time machine status prediction  
- ML-powered classification (Random Forest)  
- CSV-based historical data training  
- Interactive UI using Streamlit  
- Multiple input parameters (plant, machine, department, etc.)  
- High accuracy (~97%)  

## 🧾 Target Classes  
0: null  
1: ChangeOver  
2: No Load  
3: People  
4: Production  
5: Sample  
6: Technical  

## 🏗️ Project Structure  
Machine.ipynb  
Machine_status_code_History_New.csv  
app.py  
model_columns.pkl  
requirements.txt  
runtime.txt  
README.md  

## ⚙️ Installation  
Clone the repository:  
git clone https://github.com/your-username/machine-learn-to-talk.git  
cd machine-learn-to-talk  

Install dependencies:  
pip install -r requirements.txt  

## ▶️ Run the Application  
streamlit run app.py  

Then open: http://localhost:8501  

## 🌐 Live Demo  
https://machineerror.streamlit.app  

## 🧠 Model Details  
Algorithm: Random Forest Classifier  
Library: Scikit-learn  
Accuracy: ~97.39%  
Training Data: Historical machine status logs  

## 📥 Input Parameters  
Plant  
Technology  
Cost Center ID  
Department  
Machine ID  
Plant Shift Date  
Production Order Number  
Tool Number  

## 🛠️ Tech Stack  
Python  
Scikit-learn  
Pandas  
NumPy  
Streamlit  

## 📌 Use Cases  
Smart Manufacturing  
Predictive Maintenance  
Industry 4.0 Applications  
Machine Failure Detection  

## 🙌 Future Improvements  
Deep Learning integration  
Real-time IoT sensor data support  
Dashboard analytics  
Model retraining pipeline  
API deployment  

## 🤝 Contributing  
Contributions are welcome. Fork the repository and submit a pull request.  

## 📜 License  
This project is licensed under the MIT License.  

## 👨‍💻 Author  
1.Vedant Machindra Thorat  
2. Ashraf Shikalgar

Passionate about AI, ML & Data Science  
