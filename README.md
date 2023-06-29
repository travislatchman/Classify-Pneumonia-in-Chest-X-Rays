# Classify-Pneumonia-using-Chest-X-Rays

Data Augmentation and Visualization using X-Ray Dataset

**Description:** This project involved the use of data augmentation and visualization techniques in the classification of pneumonia vs normal X-rays.  We will learn to classify pneumonia using Chest X-Ray dataset which can be downloaded from [here](https://www.dropbox.com/s/cwvaqip12wpex6o/Lab7_XRay_chest_pnemonia.zip?dl=0). This data is already split into train, test and validation directories, so you do not have to split it yourself!  There are two sub-directories under each directory: `NORMAL` and	`PNEUMONIA`


## Implementation (see notebook for each task):
* Applied data augmentation in PyTorch to improve the model's performance.
* Visualized the model's decision-making using Grad-CAM.
* Implemented lung segmentation.
* Visualized embeddings using TSNE.


## Part 1: Train a Convolutional Neural Network (CNN) with and without data augmentation

### **`Tasks 1 and 2 `** 
Define directories for train, test, validation

### **`Task 3 `** 
Visualize four normal images and four Pneumonia images from train set. Plot class distribution of all sets using bar plots. 

![image](https://github.com/travislatchman/Classify-Pneumonia-in-Chest-X-Rays/assets/32372013/545ad3cd-cd97-4df2-a758-8617553dab6b)
![image](https://github.com/travislatchman/Classify-Pneumonia-in-Chest-X-Rays/assets/32372013/f0e2ac81-5a93-456b-9731-0a62f81f8984)
![image](https://github.com/travislatchman/Classify-Pneumonia-in-Chest-X-Rays/assets/32372013/c8a4a738-7d1b-4567-9176-e1ca23ab5ed6)
![image](https://github.com/travislatchman/Classify-Pneumonia-in-Chest-X-Rays/assets/32372013/c82abdb9-b3e5-4415-9b84-92a533f3ae9e)
![image](https://github.com/travislatchman/Classify-Pneumonia-in-Chest-X-Rays/assets/32372013/338b9a5c-56f8-4cf4-84fb-5ca85d97c524)
![image](https://github.com/travislatchman/Classify-Pneumonia-in-Chest-X-Rays/assets/32372013/221c76b2-76ac-4a00-b3f1-fe19c4e57045)
![image](https://github.com/travislatchman/Classify-Pneumonia-in-Chest-X-Rays/assets/32372013/5416c71a-1d71-470e-8e04-36e843fb5543)
![image](https://github.com/travislatchman/Classify-Pneumonia-in-Chest-X-Rays/assets/32372013/d1653be8-c04a-4a10-ad43-4d4b3d9fe549)


![image](https://github.com/travislatchman/Classify-Pneumonia-in-Chest-X-Rays/assets/32372013/ec8e2faf-85d8-4407-b13c-ff3391c07750)
![image](https://github.com/travislatchman/Classify-Pneumonia-in-Chest-X-Rays/assets/32372013/a925c967-c08c-4d14-b0b6-b1efe030157c)
![image](https://github.com/travislatchman/Classify-Pneumonia-in-Chest-X-Rays/assets/32372013/0d611110-b35e-4667-bf2d-f0cb2c35adc2)


### **`Task 4 `** 

Train a Convolutional Neural Network (CNN) using data loaders. Plot training and validation accuracy and loss in a graph. Please report the recall-rate, precision, accuracy and F1-score on test set.

![image](https://github.com/travislatchman/Classify-Pneumonia-in-Chest-X-Rays/assets/32372013/6eef74bd-f5cf-4c6e-b678-00c58808c7bc)

Recall: 0.9897, Precision: 0.7215, Accuracy: 0.7548, F1-score: 0.8346


### **`Task 5 `** 

Use data-augmentation inbuilt in Pythorch wisely. Keep all model parameters same as Task 3 for comparison.

Hint: Data augmentation in Pytorch is very simple when using dataloader. Please carefully select the augmentation parameters in `transforms.Compose`. You may want to refer to https://pytorch.org/vision/stable/transforms.html for options available.

Then train a Convolutional Neural Network (CNN) with the same parameters as Task 3. **Plot training and validation accuracy and loss in a graph.** Please report the recall-rate, precision, accuracy and F1-score on test set.

![image](https://github.com/travislatchman/Classify-Pneumonia-in-Chest-X-Rays/assets/32372013/55556f2a-2a1e-4448-99ba-ed8e2355e9d5)

Recall: 0.9615, Precision: 0.8315, Accuracy: 0.8542, F1-score: 0.8918

### **`Task 6 `** 
Compare results with and without data augmentation. Did data augmentation help improving f-score? Why do you think it worked/not worked? Display nine examples of augmented data from training set.

### Results without augmentation
Recall: 0.9897, Precision: 0.7215, Accuracy: 0.7548, F1-score: 0.8346

### Results with augmentation
Recall: 0.9615, Precision: 0.8315, Accuracy: 0.8542, F1-score: 0.8918

Comparing the results with and without data augmentation,  data augmentation has helped improve the performance of the model, particularly in terms of F1-score, precision, and accuracy.

The recall is slightly lower with data augmentation, but the precision, accuracy, and F1-score have significantly improved. This shows that the model has become more balanced in terms of its ability to correctly identify both positive and negative cases.

Data augmentation helped the model generalize better by introducing variability in the training data. By creating modified versions of the original images (e.g., through rotation, scaling, flipping, etc.), the model learns to recognize patterns and features that are invariant to these transformations. This leads to a more robust model that is better equipped to handle variations in the test data.

![image](https://github.com/travislatchman/Classify-Pneumonia-in-Chest-X-Rays/assets/32372013/8771f75a-062a-4ea2-9b4a-99f11aed9873)


## Part 2: Visualization using Grad-CAM

### **`Task 7 `** 
What is Grad-CAM? Why do you think Grad-CAM is useful?
Please give a short summary of Grad-CAM (100-200 words)

You may want to read the paper here : https://arxiv.org/abs/1610.02391

Grad-CAM (Gradient-weighted Class Activation Mapping) is a visualization technique for understanding the decisions made by convolutional neural networks (CNNs) in image classification tasks. It provides insight into which regions of the input image contribute the most to the model's prediction by generating a coarse localization map that highlights the important areas in the image. Grad-CAM can be applied to any CNN architecture without the need for architectural modifications or retraining.

Grad-CAM works by computing the gradients of the predicted class score with respect to the feature maps of the last convolutional layer. These gradients are then used to compute a weighted sum of the feature maps, resulting in a class activation map. The class activation map is then upsampled to the size of the input image to produce a heatmap that highlights the regions in the image that the model finds most relevant for the predicted class.

Grad-CAM is useful because it helps to understand and interpret the predictions made by complex CNN models. By visualizing the important regions in the input image, Grad-CAM can provide insights into the model's decision-making process, which can be helpful for model debugging, identifying biases, and improving the model's performance. Additionally, Grad-CAM can be used to generate weak localization information for tasks like object detection and semantic segmentation, where ground truth bounding boxes or segmentation masks may not be available.

### **`Task 8 `** 
Visualize 4 samples in test data (true positive, false positive, true negative, and false negative each) using Grad-CAM. For every data points, you plot the Grad-CAM image and also mention the predicted and true labels.  

![image](https://github.com/travislatchman/Classify-Pneumonia-in-Chest-X-Rays/assets/32372013/92fab7f7-f288-4ea8-a61e-b7cb9d23f3ae)
![image](https://github.com/travislatchman/Classify-Pneumonia-in-Chest-X-Rays/assets/32372013/a655fa9b-5a9c-4540-8a21-2b449b1307ef)
![image](https://github.com/travislatchman/Classify-Pneumonia-in-Chest-X-Rays/assets/32372013/39bf9134-cab8-44a3-8243-ea9983bc1d2e)
![image](https://github.com/travislatchman/Classify-Pneumonia-in-Chest-X-Rays/assets/32372013/306f3244-4ef2-4fa1-b113-14c481336a98)




## Part 3: Lung segmentation and transfer learning

### **`Task 9 `** 
Use lung segmentation as pre-processing to see if such processing helps your models. After doing lung segmentation, you will discard certain irrelevant information such as part of the skeleton, the heart, etc. Display four samples from training set to show the result of segmentations.

You can use this for segmentation: https://github.com/imlab-uiip/lung-segmentation-2d

You can also check this: https://github.com/jdariasl/COVIDNET from this paper: https://ieeexplore.ieee.org/abstract/document/9293268

**Hint 1:** In this task you need the keras.models.load_model to load the model from https://github.com/imlab-uiip/lung-segmentation-2d

**Hint 2:**You need to follow the preprocessing method in their code to segment the images, or the results will not be good.

![image](https://github.com/travislatchman/Classify-Pneumonia-in-Chest-X-Rays/assets/32372013/e735d6f0-2c80-4e02-847f-3a74570a09d7)
![image](https://github.com/travislatchman/Classify-Pneumonia-in-Chest-X-Rays/assets/32372013/549223c3-c95a-47db-ba6e-967d0ac161f9)
![image](https://github.com/travislatchman/Classify-Pneumonia-in-Chest-X-Rays/assets/32372013/eba3ee85-3ec9-4320-ae9c-e6d56a06e6c7)
![image](https://github.com/travislatchman/Classify-Pneumonia-in-Chest-X-Rays/assets/32372013/7b884a26-db6e-4962-8f64-b87b48e52650)


### **`Task 10 `** 
Repeat the previous experiments (Task5, Task 7, and Task 8) but now using lung segmentation. Compare the results and grad-cam of the new model with the model trained previously with the dataset without segmentation.

![image](https://github.com/travislatchman/Classify-Pneumonia-in-Chest-X-Rays/assets/32372013/a6e25904-451a-4edb-a4f4-06df14a91c7a)

Recall: 0.9282, Precision: 0.7686, Accuracy: 0.7804, F1-score: 0.8409

I do believe the quality of the segmentations affected the model's accuracy (it is lower compared to the model without segmentation). Specifically in the pneumonia train dataset, some of the segmentations were cutoff too much or completely dark. Also, I had to implement a try-except block for the pneumonia train dataset because U-net threw a convolution error for some images in the pneumonia (train dataset), so images that could not be segmented had to be left out. The segmentation model only had troubles for the pneumonia train dataset, but not for the normal train dataset, or the correponding normal/pneumonia in the validation or test subfolders (I could reuse the same code block just changing the paths, and didn't need try-except).

I also believe segmentations could have been better if the U-Net model was trained specifically on this dataset, and not just using pre-trained U-Net. Maybe some fine-tuning would have been helpful.

![image](https://github.com/travislatchman/Classify-Pneumonia-in-Chest-X-Rays/assets/32372013/8635b22a-3701-4906-9e43-78f0c016090a)
![image](https://github.com/travislatchman/Classify-Pneumonia-in-Chest-X-Rays/assets/32372013/494428eb-d7a9-4891-bfde-bc1026a2ce78)
![image](https://github.com/travislatchman/Classify-Pneumonia-in-Chest-X-Rays/assets/32372013/5dc9aabd-75b1-4cca-ac57-9459119f068e)
![image](https://github.com/travislatchman/Classify-Pneumonia-in-Chest-X-Rays/assets/32372013/62028cb4-adce-4e04-8792-2a04910f6004)

### Results without segmentation
Recall: 0.9615, Precision: 0.8315, Accuracy: 0.8542, F1-score: 0.8918

### Results with segmentation
Recall: 0.9282, Precision: 0.7686, Accuracy: 0.7804, F1-score: 0.8409

Regarding Grad-CAM visualizations, a comparison between the two models can help understand how each model focuses on different regions of the images to make predictions. The model trained on the non-segmented dataset may rely more on features present in the background and surrounding areas, as it has access to the entire image. In contrast, the model trained on the segmented dataset will likely focus more on the lung regions, as the segmentation process has removed most of the background and irrelevant areas.

However, a higher focus on lung regions does not necessarily guarantee better performance, as seen in the metrics above. The context provided by the background and surrounding areas in the non-segmented images might contain valuable information that helps the model achieve better results in this specific case.

Maybe, while segmentation can be helpful in some cases by isolating the region of interest and reducing the impact of background noise, it may not always lead to improved performance.

Use TSNE for visualization of embeddings obtained using model to see if embeddings form seperate clusters. Feel free to use any library.

![image](https://github.com/travislatchman/Classify-Pneumonia-in-Chest-X-Rays/assets/32372013/cd577151-7742-4e56-9ce0-b9874504e685)
