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

To simplify the task, the researcher focused exclusively on Class B and Class F fires.

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
- Located and compiled the dataset, then performed annotation on the original images for use in the research.

- Narrowed down the dataset to focus on specific fire classes to simplify the classification process.

- Composed the abstract, introduction, and methodology sections of the research paper.
