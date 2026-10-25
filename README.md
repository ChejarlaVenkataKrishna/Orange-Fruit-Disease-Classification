# Orange Fruit Disease Classification: Project Report
GreenPod Labs Data Science Assignment | Classical Computer Vision vs. Deep Learning
# 1. Executive Summary
This project builds a computer vision system that looks at a photo of an orange and classifies it into one of four conditions: fresh (healthy), blackspot, canker, or greening. Two approaches were developed and compared on the same held-out test set of 99 images:
1.Approach 1: Classical computer vision + machine learning. Hand-crafted colour (HSV, CIELAB) and texture (LBP) features fed to an SVM.
2.Approach 2: Deep learning. Transfer learning with an ImageNet-pretrained MobileNetV2.
Key result: the classical SVM reached 98.99% test accuracy (macro F1 0.9886), outperforming MobileNetV2, which reached 93.94% test accuracy (macro F1 0.9342). The SVM misclassified 1 of 99 test images; MobileNetV2 misclassified 6. Both models classified fresh and greening fruit almost perfectly. The remaining errors are concentrated in the visually similar blackspot and canker classes.
Model	Test Accuracy	Macro F1	Test Errors (of 99)
Classical SVM	0.9899	0.9886	1
MobileNetV2	0.9394	0.9342	6
# 2. Problem Statement and Objectives
Citrus diseases such as blackspot, canker and greening reduce fruit quality and market value, and manual inspection is slow and subjective. The objective is to automate visual grading of oranges into four classes from a single photograph.
The assignment requires:
•Building two distinct approaches (classical and deep learning).
•Auditing the data before modelling.
•Evaluating on a held-out test set with metrics that suit multi-class data (accuracy, macro F1, per-class precision and recall, confusion matrices).
•Comparing the approaches and demonstrating inference on a single image.
# 3. Dataset
3.1 Structure
The data is supplied as Orange Dataset.zip, which extracts into two top-level folders, train and test, each containing one sub-folder per class. The class folders are fresh, blackspot, canker, and grenning (the folder name is spelled this way in the data; it refers to greening).
3.2 Test set composition
Class	Test images
blackspot	22
canker	22
fresh	33
greening	22
Total	99
The test set is mildly imbalanced (fresh is 33% of images). For this reason macro F1 and per-class recall are reported alongside accuracy.
3.3 Dataset audit
Before modelling, the notebook audits the data by:
•Counting readable and unreadable image files (supported formats: JPG, JPEG, PNG, BMP, WEBP).
•Recording width, height and colour mode for every image.
•Plotting the class distribution by split.
•Visually spot-checking 10 random training images per class for blur, lighting differences, background differences, mislabelled samples and near-duplicates.
3.4 Observations from the visual spot-check
•Many images appear rotated or augmented, with black or white padded corners. This suggests the dataset was expanded by augmentation.
•Backgrounds differ by class. Greening samples are photographed on a uniform black background. Fresh samples mostly have white or light backgrounds, often with leaves attached. Blackspot and canker samples have mixed grey, white and black backgrounds.
•Blackspot and canker both show dark lesions on the peel, and canker lesions can be small and corky, so these two classes are the most visually similar.
Risk: because background and padding correlate with class, a model could partly learn the background rather than the disease. This matters for interpreting the very high test scores (see Section 8).
# 4. Approach 1: Classical Computer Vision + SVM
4.1 Method
Instead of learning from raw pixels, the pipeline extracts engineered features designed to capture what a human inspector looks at:
•Colour histograms in HSV space to capture overall hue and saturation (orange vs. green vs. yellow).
•Colour histograms in CIELAB space to capture perceptually uniform colour differences, useful for lesion colour.
•Local Binary Pattern (LBP) texture features to capture peel roughness and lesion texture.
The feature vector is passed to a Support Vector Machine classifier. The notebook introduction states that a Random Forest was also trained for comparison; its results are not included in the printed output reviewed here. The trained SVM is saved to /content/models/classical_svm.joblib.
4.2 Results
•Validation accuracy: 0.9732
•Test accuracy: 0.9899
•Test macro F1: 0.9886
Class	Precision	Recall	F1	Support
blackspot	1.00	0.95	0.98	22
canker	0.96	1.00	0.98	22
fresh	1.00	1.00	1.00	33
greening	1.00	1.00	1.00	22
Confusion matrix: 98 of 99 test images were classified correctly. The only error was one blackspot image predicted as canker.
# 5. Approach 2: MobileNetV2 Transfer Learning
5.1 Method
MobileNetV2 was chosen as an efficient architecture suited to resource-limited deployment. The network was initialised with ImageNet pretrained weights and fine-tuned on the orange images so it learns its own visual features rather than relying on hand-crafted ones. Training ran on CPU for 8 epochs. The saved checkpoint (/content/models/mobilenetv2.pth) stores the weights together with the class list and the normalisation mean and standard deviation, so inference is reproducible.
5.2 Training behaviour
Epoch	Train loss	Val loss	Val accuracy
1	0.8221	0.4642	0.9396
2	0.3387	0.2694	0.9664
3	0.2195	0.1896	0.9664
4	0.1706	0.1621	0.9664
5	0.1354	0.1437	0.9732
6	0.1247	0.1277	0.9799
7	0.1029	0.1167	0.9664
8	0.0921	0.1123	0.9664
Training and validation loss decrease together, with no sign of strong overfitting. Validation accuracy plateaued around 96.6 to 98.0%.
5.3 Test results
•Test accuracy: 0.9394
•Test macro F1: 0.9342
•Total test inference time: 4.36 seconds on CPU, about 44 ms per image
Class	Precision	Recall	F1	Support
blackspot	0.91	0.91	0.91	22
canker	0.90	0.82	0.86	22
fresh	0.94	1.00	0.97	33
greening	1.00	1.00	1.00	22
Confusion matrix errors (6 total):
•2 blackspot images predicted as canker
•2 canker images predicted as blackspot
•2 canker images predicted as fresh
The two canker images predicted as fresh are the most consequential mistakes, because a diseased fruit would be passed as healthy.
# 6. Model Comparison
Criterion	Classical SVM	MobileNetV2
Test accuracy	0.9899	0.9394
Macro F1	0.9886	0.9342
Weakest class	blackspot recall 0.95	canker recall 0.82
Diseased fruit predicted as fresh	0	2
Feature design	Manual (HSV, CIELAB, LBP)	Learned automatically
Training cost	Low (CPU, fast)	Higher (8 epochs of fine-tuning)
Inference	Feature extraction + SVM	About 44 ms/image on CPU
Interpretability	High: features are colour and texture statistics	Lower: learned features
Both models perfectly separate greening and handle fresh with at most minor confusion, because these classes differ strongly in colour. The difficulty lies in separating blackspot from canker, where lesion shape and texture matter more than overall colour. Here I am attaching the reference proof of one image demonstration with both models.

 <img width="733" height="299" alt="image" src="https://github.com/user-attachments/assets/5da14727-e90f-4e95-b8a4-73adc2c83407" />


# 7. Single-Image Demonstration
The notebook ends with an end-to-end demo: a user uploads one image (c (13).jpg, a canker sample) and both models predict it.
•Classical SVM prediction: canker
•MobileNetV2 prediction: canker
•MobileNetV2 class probabilities: blackspot 0.0652, canker 0.8929, fresh 0.0366, greening 0.0054
Both models agree, and MobileNetV2 reports about 89% confidence, with most of the remaining probability on blackspot, consistent with the confusion pattern above.
# 8. Discussion and Limitations
Why the SVM won. The four classes are strongly separable by colour, and the hand-crafted features directly encode colour and texture. A small dataset also favours a compact model over a deep network with millions of parameters. This outcome does not mean classical methods are generally superior: it shows they are competitive on this particular dataset.
Small test set. With 99 test images, the gap between 1 and 6 errors is only 5 images. The confidence interval around each accuracy figure is wide, so the ranking should be treated as suggestive rather than conclusive.
Possible background and augmentation shortcuts. Class-specific backgrounds (black for greening, white for fresh) and rotated, padded images may inflate scores for both models. If test images share these conditions with training images, the results will overstate real-world performance on photos taken in orchards or packing houses.
Possible augmentation leakage. If augmented copies of the same original photo appear in both train and test, the test score is optimistic. The dataset audit should be extended with a duplicate and near-duplicate check across splits.
Limited evaluation. Results come from a single train/validation/test split and a single run. Cross-validation or repeated runs would show how stable the numbers are.
Class scope. The model only recognises four conditions. Unknown conditions (other diseases, physical damage, rot) will be forced into one of the four classes.
# 9. Recommendations and Future Work
1.Check for leakage and background bias. Run perceptual-hash duplicate detection across train and test, and test on images with backgrounds the model has never seen.
2.Collect real-world test data. Photos from field and packing-house conditions with varied lighting and backgrounds.
3.Improve MobileNetV2. Train longer with data augmentation (lighting, crop, blur), unfreeze more layers, tune the learning rate, and try stronger backbones such as EfficientNet.
4.Target the blackspot/canker confusion. Add lesion segmentation or higher-resolution inputs so lesion shape and texture are captured better.
5.Prioritise recall on disease classes. In a grading setting, passing a diseased orange as fresh is costlier than a false alarm. Consider cost-sensitive training or a confidence threshold that routes uncertain images to manual inspection.
6.Consider a hybrid or ensemble. Combine hand-crafted features with deep features, or ensemble the SVM and MobileNetV2.
7.Deployment. MobileNetV2 is lightweight enough for mobile or edge devices; the SVM is also very cheap. Both models are already saved and can be wrapped in a simple inference service or app.
# 10. Conclusion
Both approaches classify orange conditions accurately. The classical HSV + CIELAB + LBP pipeline with an SVM achieved the best test performance (98.99% accuracy, macro F1 0.9886), while MobileNetV2 transfer learning reached 93.94% accuracy (macro F1 0.9342). Greening and fresh fruit are recognised nearly perfectly; blackspot versus canker is the remaining challenge. Because of the small test set and possible dataset biases, the next step should be validating on independent, real-world images before drawing firm conclusions about deployment readiness.
Appendix: Reproducibility and Notebook Notes & Answers for questions in an Assignment
•Environment: Google Colab, Python; libraries include scikit-image, OpenCV (headless), scikit-learn, joblib, PyTorch, torchvision, pandas, matplotlib and PIL. MobileNetV2 training ran on CPU.
1.In production how your model performs?
On the held-out test set, the SVM scored 98.99% accuracy (macro F1 0.9886) and MobileNetV2 scored 93.94% (macro F1 0.9342). MobileNetV2 runs at about 44 ms per image on CPU, so speed isn’t a concern. I would not expect these test numbers to carry over to production, though, for three reasons:
The dataset has class-specific backgrounds (black for greening, white for fresh) and many rotated, padded images. The models may be partly learning the background rather than the disease.
If augmented copies of the same photo appear in both train and test, the test score is optimistic.
The test set has only 99 images, so the gap between 1 error and 6 errors is weak evidence.
In real orchard or packing-house photos, with different lighting, cameras and backgrounds, I’d expect a noticeable drop. The SVM is probably more at risk, since its colour features are sensitive to lighting and white balance. I haven’t measured this, so I would validate on independent real-world images before deploying.

2. What is the drawback of your model?
Limited evaluation evidence.
The test set contains only 99 images, and results come from a single train/validation/test split and a single run. The difference between the SVM (1 error) and MobileNetV2 (6 errors) is five images, which is not statistically conclusive. Cross-validation and confidence intervals would be needed to compare the models reliably.

Limited optimisation of the deep learning model.
MobileNetV2 was fine-tuned for only 8 epochs on CPU with little hyperparameter tuning or augmentation. It is likely under-optimised, which may partly explain why it trailed the SVM. Its learned features are also harder to interpret, so it is difficult to confirm that it attends to lesions rather than backgrounds.
3.If given more time, what would you will improve? 
Collect real-world photos taken under varied lighting and backgrounds.
Check for duplicates and leakage across train and test, and test on images with unseen backgrounds.
Add data augmentation (lighting, blur, crop), train MobileNetV2 longer with more layers unfrozen, and try stronger backbones like EfficientNet.
4.Where you spend more time, during solving your assignment?
Training MobileNetV2 on CPU. This is slow, and it probably limited how many epochs you could run.
Evaluation and comparison. Interpreting confusion matrices and working out why the SVM beat the deep model.

Reference work Link:- https://colab.research.google.com/drive/1u6E424dQKmw0RTn6Ly5TzGuEzWmNaJId?usp=sharing

Reference Work Video Link:-   https://canva.link/f2mpp1z16d55lf4
