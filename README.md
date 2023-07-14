# zero-shot anomaly detection

Compare to main code which is provided in [this link](https://openaccess.thecvf.com/content/WACV2023/html/Aota_Zero-Shot_Versus_Many-Shot_Unsupervised_Texture_Anomaly_Detection_WACV_2023_paper.html), I have changed and test different parts of the algorithm such as training model, distance calculator, and etc. Which are all noted as different versions of the code in below, also the performance of each version is reported.

# changing model
The main code uses wide_resnet50_2 model for extracting needed features of each input texture image, It was mentioned in the algorithm provider's paper that they only use the model up to the 2nd layer as it can be seen in the code.

The feature extacting from model's 2nd layer was done by hook method which gathers the output of the selected layer when there is a forward function happening in that layer, so I changed the model to do the exactly the same but preventing the other layers that comes after the 2nd layer from forwarding to reduce computation time.

# changing calculation score
I have changed the calculation score function which calculates the distance between k-nearest neighbors of input image and returns the output as a heatmap.

# changing model sublayers
I have changed the model sublayers of each model layer's bottleneck (up-to 2nd layer) input and output size so it looks like the 50-layer model in the image below

<img src="images/model_example.png" alt="model_example" width="600"/>

# chaning dataset loader
I have changed the dataset loader to load input images in defualt resolution and not changing them, the point that has to be consider is that the model we are using (even if not the excat wide_resnet50_2 but the trained weights) is trained on 320*320 images so if the input is different, it can cause inaccuracy. 

|version  |Mother version |Method                       | avg. Runtime per image | avg. AUROC  | avg. Pixel AUROC |
|---------|---------------|-----------------------------|------------------------|-------------|------------------|
|#01|---------------|-----------------------------|------------------------|-------------|------------------|
|---------|---------------|-----------------------------|------------------------|-------------|------------------|
|---------|---------------|-----------------------------|------------------------|-------------|------------------|
|---------|---------------|-----------------------------|------------------------|-------------|------------------|
|---------|---------------|-----------------------------|------------------------|-------------|------------------|
|---------|---------------|-----------------------------|------------------------|-------------|------------------|

version	Mother version	Method	avg. Runtime per image	avg. AUROC	avg. Pixel AUROC
#01	-	default	1.969 s	99.50%	97.40%
#02	#01	feature extraction only up to 2nd layer	1.755 s	99.50%	97.40%
#03	#01	modified calc_score function	0.107 s	99.20%	97.10%
#04	#02	modified calc_score function	0.107 s	99.20%	97.10%
#05	#04	modified cnn model for feature extraction	0.074 s	99.20%	97.10%
#06	#04	input image size of 1024*1024 pixel	0.310 s	92.30%	-

