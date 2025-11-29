# 🔍 ChurnShield AI

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-FF6F00?style=for-the-badge&logo=scikitlearn&logoColor=white)

ChurnShield AI is a powerful customer churn prediction tool that helps businesses identify customers at risk of churning. Built with Streamlit and powered by XGBoost, it provides explainable AI insights to help you understand the factors driving churn.

## ✨ Features

- 🎯 **Accurate Predictions**: XGBoost model with 78.2% accuracy
- 📊 **Explainable AI**: SHAP values show why each prediction was made
- 🎨 **Netflix-inspired UI**: Modern, responsive interface with dark theme
- 📱 **Mobile-friendly**: Works seamlessly on all devices
- 🔍 **Interactive Visuals**: Dynamic charts and gauges for better insights

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/ARPAN58/ChurnShield-AI.git
   cd ChurnShield-AI
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app locally**
   ```bash
   streamlit run netflix_ui.py
   ```

## 🛠️ Project Structure

```
ChurnShield-AI/
├── netflix_ui.py       # Main Streamlit application (Netflix-style UI)
├── app.py             # Original Streamlit interface
├── train_model.py     # Script to train the churn prediction model
├── churn_model.pkl    # Trained XGBoost model
├── model_columns.pkl  # Feature names for the model
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## 🤖 Model Performance

| Metric          | Score |
|-----------------|-------|
| Accuracy        | 78.2% |
| Precision (Churn) | 59%  |
| Recall (Churn)   | 61%  |
| F1-Score (Churn) | 60%  |

## 🌐 Live Demo

Try the live demo on Streamlit Cloud: [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://churnshield-ai-e2g3xhzgq53fcamqqya7e9.streamlit.app/)



## 📚 How It Works

1. **Data Collection**: Input customer details through the intuitive interface
2. **Prediction**: Our XGBoost model analyzes the data in real-time
3. **Explanation**: SHAP values explain the prediction
4. **Actionable Insights**: Get recommendations to reduce churn risk

## 🛠️ Built With

- [Streamlit](https://streamlit.io/) - For building the web app
- [XGBoost](https://xgboost.ai/) - Gradient boosting framework
- [SHAP](https://shap.readthedocs.io/) - For explainable AI
- [Plotly](https://plotly.com/) - Interactive visualizations
- [Pandas](https://pandas.pydata.org/) - Data manipulation

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 📧 Contact

Your Name - ARPAN SINGH - arpansinghh2121@gmail.com

Project Link: [https://github.com/ARPAN58/ChurnShield-AI](https://github.com/ARPAN58/ChurnShield-AI)

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the amazing framework
- [XGBoost](https://xgboost.ai/) for the powerful ML algorithm
- [SHAP](https://shap.readthedocs.io/) for model interpretability
- [Font Awesome](https://fontawesome.com/) for icons

---

<div align="center">
  Made with ❤️ by Arpan singh
</div>
