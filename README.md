![Distribution plot](./ChatGPT%20Image%20Apr%2012%2C%202025%2C%2001_26_58%20PM.png)


# Breast Cancer Classification with Machine Learning Models

In this project, I built a machine learning model to classify breast cancer tumor samples as either **benign** or **malignant**. The goal is to explore how machine learning can help automate this process and optimize the model’s performance. By tuning the hyperparameters and improving accuracy, this work contributes to the potential of machine learning in healthcare.

The project is split into three parts:

### **Part 1: Data Exploration and Model Training**
In Part 1, I dive into the breast cancer dataset, clean up the data, and then train a **kNN classifier** to predict tumor classifications. I experiment with different values for **k** (the number of neighbors) and evaluate how the model performs on a test dataset. The aim here is to find the best **k** that gives us the highest accuracy and better understand the dataset’s features.

### **Part 2: Model Optimization and Prediction**
In Part 2, I take the model from Part 1 and optimize it using **grid search**. By searching for the best hyperparameters like **k** and the **weights** for neighbors (uniform or distance-based), I aim to push the model’s accuracy even further. Once I’ve found the best settings, I use the optimized model to classify new tumor samples from the `aim` dataset. Finally, I measure the model's performance on the test set and make predictions for unseen data.

### Part 3: Classification of Breast Cancer Cell Samples with kNN, Logistic Regression, and Random Forest
In Part 3, I compare the performance of three supervised machine learning algorithms — **k-Nearest Neighbors (kNN)**, **Logistic Regression**, and **Random Forest** — for classifying breast cancer cell samples as benign or malignant. The objective is to identify the model that best detects malignant tumors with a high recall rate, as this is critical in a medical diagnostic context. I preprocess and train the models using the provided training data, evaluate their performance on a separate test dataset, and finally apply the best-performing classifier to predict the tumor types in an unseen target dataset.

### Part 4: Modeling and Evaluation of a Deep Feedforward Artificial Neural Network (ANN)
In Part 4, I instantiating a deep feedforward artificial neural network (ANN) using TensorFlow/Keras. The model contains five hidden layers, each with 50 artificial neurons, and an output layer with a single neuron for binary classification (benign or malignant). Regularization is applied with a 30% dropout rate to prevent overfitting. The model is trained, and learning curves are plotted to visualize performance.
Additionally, early stopping is implemented to halt training if the model’s performance does not improve after two epochs, thereby reducing the risk of overfitting. The results are compared between the regularized model (with dropout) and the early-stopped model, revealing that both models achieved the same validation accuracy of 0.9816, suggesting both approaches are effective in preventing overfitting and performing well on unseen data.

---

**Key Highlights**

- **Data Prep:** Loading and exploring the breast cancer dataset, cleaning the data, and preparing it for model training. This step includes checking for missing values, encoding labels, and standardizing features.
- **Training:** Building and training three classifiers: k-Nearest Neighbors (kNN), Logistic Regression, and Random Forest on labeled tumor data.
- **Evaluation:** Evaluating the performance of the three models on a separate test dataset. Focus is given to recall, accuracy, and precision, with particular emphasis on recall to ensure the best detection of malignant tumors.
- **Optimization:** Using grid search to fine-tune the kNN classifier’s hyperparameters, such as the number of neighbors (k) and how neighbors are weighted (uniform vs distance). Similarly, optimizing hyperparameters for Logistic Regression and Random Forest.
- **Model Comparison:** Comparing the performance of kNN, Logistic Regression, and Random Forest using metrics like accuracy, precision, and recall. The Random Forest model was found to be the best performer, with the highest recall rate for malignant tumor detection.
- **Prediction:** Using the best-performing model (Random Forest) to predict tumor classifications on the aim dataset, even when the class label is missing. These predictions are stored in the class_pred column.
- **Final Insights:** The Random Forest model outperforms both kNN and Logistic Regression in detecting malignant tumors, making it the most reliable for this medical application.

**What’s in the Data?**
The dataset includes various features related to tumor cell characteristics, such as:

- Clump Thickness
- Uniformity of Cell Size
- Uniformity of Cell Shape
- Marginal Adhesion
- Bare Nuclei
- Bland Chromatin
- Mitoses
- Class (Benign = 0, Malignant = 1)

**The task?** To predict whether a given tumor is benign (0) or malignant (1) based on these features.

**Model Optimization & Insights**
In **Part 2**, I optimized the kNN classifier by using grid search to explore:

The number of neighbors (n_neighbors).

How the neighbors are weighted (uniform vs distance).

**Part 3** involved comparing three models: kNN, Logistic Regression, and Random Forest. After evaluating the models, Random Forest was identified as the best model, with the highest recall for malignant tumor detection. It also outperformed kNN and Logistic Regression in terms of accuracy and precision. The best Random Forest model was used to predict tumor malignancy in the aim dataset, even when the class label was missing. These predictions were stored in the class_pred column.

In **Part 4**, a deep feedforward ANN was evaluated with dropout regularization and early stopping. The results show that both methods effectively prevented overfitting, with an identical validation accuracy of 0.9816.

**How It Works**

- Load the Data: Load the training, test, and aim datasets, ensuring proper data cleaning and preprocessing.
- Build the Models: Create three classifiers (kNN, Logistic Regression, and Random Forest), including steps for data standardization and classification.
- Train & Optimize: Train the models and fine-tune hyperparameters using grid search to optimize each algorithm's performance.
- Make Predictions: Use the best-performing model (Random Forest) to predict tumor classifications on the aim dataset.
- Assess Performance: Evaluate accuracy, precision, recall, and confusion matrices to determine the best model.

**The Results**
By the end of the project, Random Forest emerged as the most reliable model for classifying breast cancer tumors, with the highest recall rate for detecting malignant tumors. This project demonstrates how machine learning can be leveraged in healthcare to automate tumor classification and potentially aid in early cancer detection.

**Tech Stack**
- Jupyter Notebooks (for interactive development)
- Python (the go-to language for data science)
- Pandas (for data manipulation)
- Scikit-learn (machine learning and grid search)
- Matplotlib/Seaborn (for visualizing results)

