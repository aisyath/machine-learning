import os
import sys
import numpy as np
import pandas as pd
import pickle
from collections import Counter
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, recall_score, 
                             f1_score, roc_auc_score)
from sklearn.model_selection import (train_test_split, cross_val_score, cross_val_predict, 
                                       StratifiedKFold)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch

# Load the dataset
DATASET_PATH = 'C:/Users/ACER/OneDrive - mail.unnes.ac.id/katalis/Crop_recommendation.csv'
if not os.path.exists(DATASET_PATH):
    print(f"File not found at: {DATASET_PATH}")
    sys.exit()
crop_data = pd.read_csv(DATASET_PATH)

# Prepare the dataset
y = crop_data['label']
x = crop_data.drop(['label'], axis=1)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Feature scaling
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Hierarchical Clustering
hc = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
clusters = hc.fit_predict(x_scaled)

# Modular function to save model
def save_model(model, model_name):
    model_filename = f'C:/Users/ACER/OneDrive - mail.unnes.ac.id/katalis/{model_name}_model.pkl'
    with open(model_filename, 'wb') as model_file:
        pickle.dump(model, model_file)
    print(f"{model_name} has been saved to {model_filename}")

# Modular function to plot confusion matrix
def plot_confusion_matrix(conf_matrix, model_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.tight_layout()
    plot_filename = os.path.join(output_dir, f'{model_name}_confusion_matrix.png')
    plt.savefig(plot_filename)
    plt.close()
    print(f"Confusion matrix for {model_name} saved as {plot_filename}")

# Modular function to visualize clustering results
def plot_dendrogram(x_scaled):
    plt.figure(figsize=(10, 7))
    dendrogram = sch.dendrogram(sch.linkage(x_scaled, method='ward'))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Samples')
    plt.ylabel('Distance')
    plt.show()

plot_dendrogram(x_scaled)

# Analyze clustering results
cluster_df = pd.DataFrame({'Cluster': clusters, 'Label': y})
for i in range(4):
    print(f"Crops in Cluster {i}: {cluster_df[cluster_df['Cluster'] == i]['Label'].unique()}")
    print("---------------------------------------------------------------")

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_encoded, test_size=0.2, random_state=42)

# Define models
models = {
    "KNN": KNeighborsClassifier(),
    "DT": DecisionTreeClassifier(),
    "RFC": RandomForestClassifier(),
    "GBC": GradientBoostingClassifier(),
    "XGB": XGBClassifier(),
    "SVM": SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42),
    "ANN": MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=500, learning_rate_init=0.001, 
                         early_stopping=True, random_state=42)
}

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Store cross-validation results
cv_results = {}
confusion_matrices = {}
model_predictions = {}
model_accuracies = {}

# Create output directory
output_dir = 'C:/Users/ACER/OneDrive - mail.unnes.ac.id/katalis/confusion_matrices/'
os.makedirs(output_dir, exist_ok=True)

# Train, evaluate, and pickle each model
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")

    # Cross-validation scores
    cv_scores = cross_val_score(model, x_scaled, y_encoded, cv=cv, scoring='accuracy')
    mean_accuracy = np.mean(cv_scores)
    std_dev = np.std(cv_scores)

    print(f"{model_name} Mean Accuracy: {mean_accuracy:.4f}")
    print(f"{model_name} Standard Deviation: {std_dev:.4f}")

    # Cross-validation predictions
    y_pred = cross_val_predict(model, x_scaled, y_encoded, cv=cv)
    conf_matrix = confusion_matrix(y_encoded, y_pred)
    confusion_matrices[model_name] = conf_matrix

    print(f"\nConfusion Matrix for {model_name}:")
    print(conf_matrix)

    # Plot confusion matrix
    plot_confusion_matrix(conf_matrix, model_name)

    # Store results for comparison
    cv_results[model_name] = {
        "mean_accuracy": mean_accuracy,
        "std_dev": std_dev,
        "confusion_matrix": conf_matrix
    }

    # Train the model on the full training set and predict on the test set
    model.fit(x_train, y_train)
    y_test_pred = model.predict(x_test)

    # Decode predictions
    y_test_pred_decoded = label_encoder.inverse_transform(y_test_pred)
    y_test_decoded = label_encoder.inverse_transform(y_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test_decoded, y_test_pred_decoded)
    print(f"{model_name} Accuracy on Test Set: {accuracy:.4f}")

    # Store predictions and accuracies
    model_predictions[model_name] = y_test_pred
    model_accuracies[model_name] = accuracy

    # Precision, Recall, F1-Score
    precision = precision_score(y_test_decoded, y_test_pred_decoded, average='weighted')
    recall = recall_score(y_test_decoded, y_test_pred_decoded, average='weighted')
    f1 = f1_score(y_test_decoded, y_test_pred_decoded, average='weighted')

    print(f"{model_name} Precision: {precision:.4f}")
    print(f"{model_name} Recall: {recall:.4f}")
    print(f"{model_name} F1-Score: {f1:.4f}")

    # ROC-AUC (if model supports probability estimation)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(x_test)
        roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average='weighted')
        print(f"{model_name} ROC-AUC: {roc_auc:.4f}")
    else:
        print(f"{model_name} does not support probability estimation for ROC-AUC.")

    # Save the model
    save_model(model, model_name)

# Voting mechanism to ensemble the predictions
def ensemble_voting(predictions_dict, accuracies_dict):
    final_predictions = []
    for i in range(len(x_test)):
        current_predictions = [pred[i] for pred in predictions_dict.values()]
        prediction_counter = Counter(current_predictions)
        most_common_prediction, count = prediction_counter.most_common(1)[0]

        if count > 4:
            final_predictions.append(most_common_prediction)
        elif count == 3 and len(prediction_counter) == 3:
            top_two_predictions = prediction_counter.most_common(2)
            best_prediction = max(top_two_predictions, key=lambda x: accuracies_dict[x[0]])
            final_predictions.append(best_prediction[0])
        elif count == 2 and len(prediction_counter) == 4:
            top_three_predictions = prediction_counter.most_common(3)
            best_prediction = max(top_three_predictions, key=lambda x: accuracies_dict[x[0]])
            final_predictions.append(best_prediction[0])
        else:
            final_predictions.append(most_common_prediction)

    return final_predictions

# Run ensemble voting and decode predictions
final_predictions = ensemble_voting(model_predictions, model_accuracies)
final_predictions_decoded = label_encoder.inverse_transform(final_predictions)

# Compare with actual y_test values (decoded)
y_test_decoded = label_encoder.inverse_transform(y_test)

# Evaluate ensemble predictions
ensemble_accuracy = accuracy_score(y_test_decoded, final_predictions_decoded)
print(f"\nEnsemble Model Accuracy: {ensemble_accuracy:.4f}")

# Display example output
ensemble_output = final_predictions_decoded[0]  # Example for the first test instance
print(f"Ensemble Output: {ensemble_output}")

# Save the ensemble output
with open('C:/Users/ACER/OneDrive - mail.unnes.ac.id/katalis/ensemble_output.pkl', 'wb') as output_file:
    pickle.dump(final_predictions_decoded, output_file)

print("Ensemble model output has been saved.")
