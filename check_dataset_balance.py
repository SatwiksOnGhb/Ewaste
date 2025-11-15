from sklearn.utils import class_weight
import numpy as np

import os

# Update the paths if your folder names are different
e_count = len(os.listdir('data/e-waste'))
ne_count = len(os.listdir('data/non-e-waste'))

print(f"E-WASTE images: {e_count}")
print(f"NON-E-WASTE images: {ne_count}")

# Calculate class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_data.classes),
    y=train_data.classes
)
class_weights = dict(enumerate(class_weights))

print("Class Weights:", class_weights)
