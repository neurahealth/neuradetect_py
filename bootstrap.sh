#!/bin/bash

sudo apt-get update

sudo apt-get install python3-venv -y
python3 -m venv venv_name
source venv_name/bin/activate

pip install --upgrade pip
sudo apt install git-all -y

git clone https://github.com/neurahealth/neuradetect_py.git
cd neuradetect_py
pip install -r requirements.txt

git clone https://github.com/nyukat/breast_density_classifier.git
git clone https://github.com/nyukat/BIRADS_classifier.git
git clone https://github.com/nyukat/breast_cancer_classifier.git
cd breast_cancer_classifier
git clone https://github.com/nyukat/GMIC.git

cd ..

cp density_model_torch_modified.py breast_density_classifier
cp birads_prediction_torch_modified.py BIRADS_classifier
cp crop_mammogram_extract_centers.py breast_cancer_classifier
cp run_model.py breast_cancer_classifier
cp exam_list_before_cropping_smaller.pkl breast_cancer_classifier
cp run_model_gmic.py breast_cancer_classifier/GMIC

gsutil cp gs://bucket-name/credential_file.json .
cp exam_list_before_cropping_smaller.pkl breast_cancer_classifier/sample_data

sudo -apt get update install tmux
tmux

#activate environment
cd neuradetect_py/breast_density_classifier
gsutil cp gs://bucket-name/credential_file.json .
python density_model_torch_modified.py cnn

#open new shell and activate environment
tmux
cd neuradetect_py/BIRADS_classifier
python birads_prediction_torch_modified.py

#open new shell and activate environment
tmux
cd neuradetect_py/breast_cancer_classifier
python crop_mammogram.py

#open new shell and activate environment
tmux
cd neuradetect_py/breast_cancer_classifier
python run_model.py

#open new shell and activate environment
tmux
cd neuradetect_py/breast_cancer_classifier/GMIC
mkdir sample_output
python run_model_gmic.py
