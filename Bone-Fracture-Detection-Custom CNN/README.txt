// This project aims to create a ML model using custom CNN, which can analyze bone fracture of wider ranges than just Wrist bones. 

However, it lacks sufficient training data on skull, ribcage and Pelvis. So, further quality dataset is required, in order to detect fractures from those regions
properly.

// Multiple Datasets are combined here in order to get a wide pool of training data from various parts of the human body.

1. FracAtlas Dataset - https://www.kaggle.com/datasets/akshayramakrishnan28/fracture-classification-dataset
2. Bone Fracture Dataset (Tibia and Fibula) - https://www.kaggle.com/datasets/orvile/bone-fracture-dataset
3. Bone Fracture Multi-Region X-ray Data - https://www.kaggle.com/datasets/bmadushanirodrigo/fracture-multi-region-x-ray-data

In order to prevent overfitting due to duplicate data across these datasets, duplication removal is done in the model during the data preprocessing stage.
