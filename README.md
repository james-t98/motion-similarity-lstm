# 🧠 AI-Based Movement Quality Assessment [motion-similarity-lstm]

This project explores the use of AI to evaluate human movement technique. Using pose estimation and time-series modeling, we assess how closely a person’s movement (e.g. a squat) matches a reference motion. The system enables automated performance feedback and benchmarking based on movement similarity.

---

## 🔍 Supported Similarity Methods

| Method     | Description                                 |
|------------|---------------------------------------------|
| `cosine`   | Cosine similarity between angle windows     |
| `mse`      | Mean squared difference between windows     |
| `dtw`      | Dynamic Time Warping on time series         |

---

## 📚 Technologies Used

- Python 3.11
- TensorFlow / Keras
- MediaPipe & OpenCV
- NumPy, SciPy, pandas
- scikit-learn
- seaborn, matplotlib
- DTW (Dynamic Time Warping)

---

## 🎯 Research Goals

- Analyze squat technique using pose-based features
- Benchmark movement similarity with different metrics
- Train LSTM to predict similarity scores
- Provide meaningful visual and statistical analysis
- Lay the foundation for real-time and generative feedback

---

## 🔮 Future Work

- Extend to other exercises (e.g. sprinting, football kicks)
- Integrate real-time webcam input
- Add VLMs for motion feedback generation
- Support long-term user performance tracking
- Deploy as an interactive web application

---

## 👤 Author

**Jaime Tellie**  
*MSc Data Analytics Candidate*  
Thesis Project – 2025

---

## 📄 License

This project is licensed under the MIT License. Feel free to use, adapt, or build on it with proper credit.

---

## 💬 Feedback or Collaboration?

Open to research partnerships, feedback, or extending the system into new sports, rehab, or performance domains.  
Feel free to fork, raise issues, or get in touch!
