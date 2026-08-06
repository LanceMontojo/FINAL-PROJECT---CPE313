<h1 align="center">🔥 Fire Classification System with Suppression Guidance using Deep Learning 🔥</h1>
<br>
<p align="center">
  <img src="https://hackster.imgix.net/uploads/attachments/1411767/ezgif-1-f6bfc1e1fb.gif?auto=format%2Ccompress&gifq=35&w=400&h=300&fit=min" width="500" alt="Fire Classification Animation"/>
</p>
<br>

## Dataset and Preprocessing
<p align="left">
The dataset used in this study was derived from a previously published work that compiled data from five existing fire-related datasets. These datasets were obtained from the following source: https://www.kaggle.com/datasets/imankhammash/classesoffire. The raw images were processed and annotated using the Roboflow platform.

During preprocessing, the images were automatically oriented and resized. Data augmentation techniques such as flipping, rotation, cropping, and shearing were applied. However, augmentations involving color changes, including grayscale conversion, hue adjustment, and saturation modification, were deliberately avoided. This decision was made because color plays an essential role in accurately classifying different types of fire.

To simplify the classification task and align the research scope, this study focuses exclusively on two fire categories: Class B (flammable liquids) and Class F (cooking oils and fats). Images belonging to other fire classes were intentionally excluded from the dataset and evaluation.

## Models

Three models were utilized:
- YOLOv11 was chosen for its real-time object detection capabilities and high efficiency in detecting multiple fire types in a single pass.

- RetinaNet was selected due to its strong performance in handling class imbalance through focal loss, which is beneficial for underrepresented fire classes.

- RT-DETR was used for its ability to model global context using transformers, improving detection accuracy in complex fire scenes.

## Metrics
The evaluation metrics employed in this study are Average Precision at IoU threshold 0.50 (AP50) and Mean Average Precision across IoU thresholds from 0.50 to 0.95 (mAP50–95), which provide a comprehensive assessment of model performance in both lenient and strict detection scenarios.

## Testing
To evaluate the model's performance in a practical setting, the best-performing model, RT-DETR, was deployed using Streamlit. Users can upload an image or video containing Class B or Class F fires, and the system outputs the predicted fire class, which is then compared against the actual class for validation.

Link for the Streamlit App: https://final-project---cpe313-dtkebcnukh5khdv97cmrbc.streamlit.app/

## Contribution
- Designed and implemented the end-to-end data engineering pipeline, including dataset collection, integration, preprocessing, augmentation, and annotation for model training.

- Developed, trained, evaluated, and deployed the deep learning models and the Streamlit application for real-time fire detection and extinguisher recommendation.

- Composed the abstract, introduction, and methodology sections of the research paper.

## Setup

The easiest way to try the project is through the deployed Streamlit application: https://final-project---cpe313-dtkebcnukh5khdv97cmrbc.streamlit.app/
 
1. Open the application.
2. Choose an input type (Image or Video).
3. Upload an image or video containing a Class B or Class F fire.
4. Click Run Detection.
5. The application will:
    - Detect the fire type using the trained RT-DETR model.
    - Display the detection results with bounding boxes.
    - Identify whether the fire is Class B or Class F.
    - Recommend the appropriate extinguisher and indicate extinguishers that should be avoided.

### Example Output

After clicking **Run Detection**, the application displays the detected fire class with bounding boxes and provides the corresponding fire extinguisher recommendation.

Below are examples of the expected results:

Class B:
<p align = "center"> 
  <img width="1111" height="767" alt="image" src="https://github.com/user-attachments/assets/e6c2e92e-8457-47a5-9e6c-941e13be4d29" />
</p>

Class F:
<p align = "center"> 
  <img width="1190" height="862" alt="image" src="https://github.com/user-attachments/assets/0952d32a-526a-4f01-98c1-a8f842f6a908" />
</p>
