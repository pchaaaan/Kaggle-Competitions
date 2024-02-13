<h1>Santander Customer Satisfaction Prediction</h1>

<p>This project aims to predict customer satisfaction for Santander Bank using a dataset provided by Kaggle. The goal is to identify unsatisfied customers early on, allowing the bank to take proactive steps to improve their satisfaction levels. I used a Decision Tree Classifier, leveraging Python and its powerful libraries like Pandas, NumPy, Scikit-learn, Seaborn, and Matplotlib for data manipulation, model building, and visualization.</p>

<h2>Project Structure</h2>

<ul>
  <li><strong>Data Preprocessing</strong>: Loaded and examined the training and testing datasets using Pandas. The training set consists of 76020 entries with 371 features, while the testing set contains 75818 entries with 370 features. We performed data cleaning to handle missing values and explored the datasets to understand the data distribution.</li>
  <li><strong>Exploratory Data Analysis (EDA)</strong>: Utilized Seaborn and Matplotlib for visualizing the distribution of the target variable and understanding the data structure. This helped in identifying patterns and potential features for the prediction model.</li>
  <li><strong>Feature Engineering</strong>: Prepared the data for modeling by separating features and the target variable. The datasets were split into training and testing sets to evaluate the model's performance.</li>
  <li><strong>Model Building</strong>: Trained a Decision Tree Classifier on the training data. The model was then used to make predictions on the testing data.</li>
  <li><strong>Evaluation</strong>: Assessed the model's accuracy and computed a confusion matrix to evaluate its performance in classifying the customer satisfaction levels. Further analysis was conducted through classification reports, providing insights into precision, recall, and f1-score.</li>
  <li><strong>Visualization</strong>: Plotted the Decision Tree to visualize the decision-making process and understand how the model arrived at its predictions.</li>
  <li><strong>Prediction and Submission</strong>: Made predictions on the unseen test dataset and prepared a submission file formatted for Kaggle competition submission.</li>
</ul>

<h2>Technologies Used</h2>

<ul>
  <li>Python</li>
  <li>Pandas for data manipulation</li>
  <li>NumPy for numerical computations</li>
  <li>Scikit-learn for model building and evaluation</li>
  <li>Seaborn and Matplotlib for data visualization</li>
  <li>Decision Tree Classifier for prediction</li>
</ul>

<h2>How to Run</h2>

<ol>
  <li>Clone this repository.</li>
  <li>Ensure you have all the required libraries installed.</li>
  <li>Load the datasets (make sure to replace the dataset paths with your local paths).</li>
  <li>Run the Jupyter Notebook/Python script to train the model and make predictions.</li>
</ol>

<h2>Results</h2>

<p>The Decision Tree Classifier achieved an accuracy of approximately 92.29% on the test set, demonstrating its effectiveness in predicting customer satisfaction levels. Detailed performance metrics can be found in the evaluation section of the notebook.</p>

<h2>Conclusion</h2>

<p>This project demonstrates the use of a Decision Tree Classifier to predict customer satisfaction for Santander Bank. Through careful data preprocessing, exploratory data analysis, and model evaluation, I was able to achieve promising results. Future work could involve experimenting with other machine learning models and techniques to improve prediction accuracy.</p>
