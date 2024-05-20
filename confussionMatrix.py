import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn.metrics import confusion_matrix

# Sample data
desvest = np.array([5.04392E-05, 6.04415E-05, 0.000184638, 4.78467E-05, 3.70054E-05, 0.000111052,
                    8.72067E-06, 0.000103816, 5.95949E-05, 6.11645E-05, 0.000178674, 0.000310591,
                    0.000140427, 2.52131E-05, 1.555102102, 9.4332E-05])

TOF_values = np.array([0.0016211, 0.0017334, 0.0015821, 0.0014258, 0.0016162, 0.0019434, 0.0019688,
                       0.0026221, 0.0015938, 0.0016094, 0.0023282, 0.001919, 0.002334, 0.001709,
                       0.0016993, 0.0014844])

density_values = np.array([1.719206353, 1.584155443, 1.681393835, 1.698312575, 1.738761829, 1.624408437, 1.655708519,
                           1.715820696, 1.680350877, 1.701151368, 1.593159483, 1.735227626, 1.675895509, 1.705817017,
                           1.555102102, 1.72366369])

# Calculate the adjusted TOF values
TOF_values_adjusted = TOF_values - 2 * desvest

print("Adjusted TOF values:", TOF_values_adjusted)


# Define thresholds
TOF_threshold = 0.00155809  # TOF from 95% trust plot

density_threshold = 1.61337226015078 #25 Strikes Mod



# Classify based on thresholds
TOF_classified = TOF_values_adjusted < TOF_threshold
density_classified = density_values > density_threshold
print(density_classified)
print(TOF_classified)

# Create labels for the confusion matrix
true_labels = density_classified
pred_labels = TOF_classified

# Compute the confusion matrix
cm = confusion_matrix(pred_labels,true_labels)

# Convert counts to percentages
cm_percentages = cm / cm.sum() * 100

# Combine counts and percentages for annotations
labels = np.array([f'{count}\n({percent:.2f}%)' for count, percent in zip(cm.flatten(), cm_percentages.flatten())]).reshape(2, 2)

# Set font properties
font_properties = {'family': 'Times New Roman', 'weight': 'bold', 'size': 12}
rcParams.update({'font.family': 'Times New Roman', 'font.weight': 'bold', 'font.size': 12})

# Visualize the confusion matrix
plt.figure(figsize=(3.5433, 2.3622))
sns.heatmap(cm_percentages, annot=labels, fmt='', cmap='Blues', xticklabels=['False', 'True'], yticklabels=['False', 'True'], cbar=False)
plt.xlabel('Density Check')
plt.ylabel('TOF SSW Check')

# Save the plot in high resolution
plt.savefig('confusion_matrix_ssw.png', dpi=600, bbox_inches='tight')

# Display the plot
plt.show()