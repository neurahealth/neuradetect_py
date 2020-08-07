

# neuradetect_py has four AI models (Breast Density Classifier, BIRADS Classifier, Breast Cancer Classifier, GMIC) which will give predictions and visualization by using tensorflow and PyTorch.

This file covers all requirments and dependencies from scratch for application neuradetect_py

## follow the below steps to run models

### Step 1
#### Create pubsub topic and subscription
Go to Google Cloud Platform \
Click on Topics, create a new topic \
create a subscription for the topic \
Click on Subscriptions, select topic and create subscription \
In Google Cloud Platform select storage click on browser and create bucket to store artifacts eg. credential file etc \
create another bucket to store logs.
### Note 
- Create two topic to run all models parallelly
first topic will process 3 scripts and publish message for second topic to process another 2 scripts
create three subscriptions of first topic and two subscriptions of second topic
  - [x] Topic-1
    - Topic-1-subscription-1
    - Topic-1-subscription-2
    - Topic-1-subscription-3
  - [x] Topic-2
    - Topic-2-subscription-1
    - Topic-2-subscription-2
        
#### Create a vm instance and run commands on SSH terminal window
Recommendation is to have N1 8vcpu configuration to process all models smoothly

### Step 2
Follow steps in sequence from [bootstrap.sh](https://github.com/neurahealth/neuradetect_py/blob/master/bootstrap.sh "bootstrap.sh") & install all dependencies 

### Application flow diagram

<img src="Application%20flow.png">

## How backend of NeuraDetect work ?
- From original model we modified the model script, totally there are five scripts [density_model_torch_modified.py](https://github.com/neurahealth/neuradetect_py/blob/master/density_model_torch_modified.py "density_model_torch_modified.py"), [birads_prediction_torch_modified.py](https://github.com/neurahealth/neuradetect_py/blob/master/birads_prediction_torch_modified.py "birads_prediction_torch_modified.py"),
[crop_mammogram_extract_centers.py](https://github.com/neurahealth/neuradetect_py/blob/master/crop_mammogram_extract_centers.py "crop_mammogram_extract_centers.py"), [run_model.py](https://github.com/neurahealth/neuradetect_py/blob/master/run_model.py "run_model.py"), [run_model_gmic.py](https://github.com/neurahealth/neuradetect_py/blob/master/run_model_gmic.py "run_model_gmic.py") from four different repository's. All scripts have to be run separately.
- when user will upload images from front end then message with stoarge information of images will get generated and sent to pubsub topic-1. 
- Subscriptions of topic-1 one will run scripts (density_model_torch_modified.py, birads_prediction_torch_modified.py, crop_mammogram_extract_centers.py) accordingly and crop_mammogram_extract_centers.py will pass message information in pubsub tpoic-2 and subscription of that topic-2 will process remaining two script( run_model.py, run_model_gmic.py). \
last two scripts are dependend on (crop_mammogram_extract_centers.py) script.


#### Models

Model Name                | Repository
--------------------------| -------------
Breast Density Classifier | [Model](https://github.com/nyukat/breast_density_classifier "Model")
BIRADS Classifier         | [Model](https://github.com/nyukat/BIRADS_classifier "Model")
Breast Cancer Classifier  | [Model](https://github.com/nyukat/breast_cancer_classifier "Model")
GMIC                      | [Model](https://github.com/nyukat/GMIC "Model")
