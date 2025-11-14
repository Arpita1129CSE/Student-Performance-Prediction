## 🎓 Student Performance Prediction System

### 📘 Overview

The **Student Performance Prediction System** is a Machine Learning project designed to estimate students’ final grades based on multiple academic, social, and behavioral factors such as attendance, study time, assignments, and stress levels.
It helps educators and institutions identify at-risk students early and understand key contributors to academic success.

---

### 🧠 Objective

To build a predictive model that can:

* Analyze student data (attendance, scores, participation, etc.)
* Predict their **grade (A–E)** using classification
* Visualize relationships between features and performance

---

### 🧩 Features Used

| Category       | Features                                                                                                   |
| -------------- | ---------------------------------------------------------------------------------------------------------- |
| **Academic**   | Midterm_Score, Final_Score, Assignments_Avg, Quizzes_Avg, Participation_Score, Projects_Score, Total_Score |
| **Behavioral** | Attendance (%), Study_Hours_per_Week, Stress_Level (1-10), Sleep_Hours_per_Night                           |
| **Social**     | Extracurricular_Activities, Internet_Access_at_Home, Parent_Education_Level                                |

---

### 🛠️ Technologies Used

* **Python**
* **Pandas** – Data handling
* **NumPy** – Numerical computation
* **Matplotlib / Seaborn** – Data visualization
* **Scikit-learn (sklearn)** – Machine learning (Random Forest Classifier, Label Encoding, Scaling)

---

### ⚙️ Steps Involved

#### 1️⃣ Data Preprocessing

* Removed unnecessary columns: `studentid`, `firstname`, `lastname`, `email`, `gender`, `age`, `family income`, `department`
* Handled missing values
* Encoded categorical variables
* Scaled numeric data using `StandardScaler`

#### 2️⃣ Model Training

* Split data into **training (80%)** and **testing (20%)**
* Used **RandomForestClassifier** for training

#### 3️⃣ Model Evaluation

* Evaluated using:

  * Accuracy Score
  * Classification Report
  * Confusion Matrix (visualized using Seaborn heatmap)

#### 4️⃣ Visualization

* **Correlation heatmap**
* **Final Score vs Total Score by Grade (scatter plot)**
* Additional relationship plots like:

  * Study Hours vs Total Score
  * Stress Level vs Grade

#### 5️⃣ Prediction

You can input new student data and get the **predicted grade**:

```python
sample = {
    'Attendance (%)': 90,
    'Midterm_Score': 75,
    'Final_Score': 80,
    'Assignments_Avg': 70,
    'Quizzes_Avg': 65,
    'Participation_Score': 80,
    'Projects_Score': 85,
    'Total_Score': 75,
    'Study_Hours_per_Week': 15,
    'Extracurricular_Activities': 1,
    'Internet_Access_at_Home': 1,
    'Parent_Education_Level': 2,
    'Stress_Level (1-10)': 5,
    'Sleep_Hours_per_Night': 7
}

print("Predicted Grade:", predict_grade(sample))
```

✅ Example Output:

```
Predicted Grade: C
```

---

### 📊 Sample Visualizations

* Confusion Matrix Heatmap
* Correlation Heatmap
* Final Score vs Total Score by Grade (scatter plot)
* Study Hours vs Total Score (color-coded by Grade)

---

### 📈 Results

* **Model Used:** Random Forest Classifier
* **Accuracy:** ~85–90% (depending on dataset)
* **Grades Predicted:** A, B, C, D, E

---

### 🧮 Example Predictions

| Sample | Description                      | Predicted Grade |
| ------ | -------------------------------- | --------------- |
| 1      | High performer (90+ total score) | A               |
| 2      | Average student                  | C               |
| 3      | Weak performer                   | D/E             |
| 4      | Hardworking but stressed         | B/C             |
| 5      | Low attendance but smart         | B               |

---

### 💡 Future Enhancements

* Add deep learning model (Neural Network)
* Integrate web interface (Flask/Streamlit)
* Include time-series analysis to track progress over semesters

