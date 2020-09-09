"""
The Originalfile belongs to breast_cancer_classifier under GNU Affero General Public License v3.0
and is available here: https://github.com/nyukat/breast_cancer_classifier
Neura Health made modifications to fils to run with neuradetect application.
"""

import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import json
import sys
import firebase_admin
from firebase_admin import firestore,storage,credentials
from google.cloud import pubsub
from google.cloud import storage as store
import logging.handlers
import png
import pydicom
import os
import dotenv
dotenv.load_dotenv()
from PIL import Image
from resizeimage import resizeimage
from numpy import asarray
import shutil
import csv

import argparse
import collections as col
import numpy as np
import os
import pandas as pd
import torch
import tqdm

import src.utilities.pickling as pickling
import src.utilities.tools as tools
import src.modeling.models as models
import src.data_loading.loading as loading
from src.constants import VIEWS, VIEWANGLES, LABELS, MODELMODES

class PubsubMessageHandler():
    def PubsubCallback(self,message):
        msg_id =  message.message_id
        #print(msg_id)

        filename = msg_id+'.log'
        my_logger = logging.getLogger(msg_id)
        my_logger.setLevel(logging.INFO)
        handler = logging.handlers.RotatingFileHandler(filename, maxBytes=20)
        my_logger.addHandler(handler)

        sub_data = message.data.decode('utf-8')
        d = json.loads(sub_data)

        userId = d['userId']         # userid is used to create path to store results for same user
        #print("User Id : ",userId)

        path = d['path']

        url = d['url']               # url is using to copy images from firestore to root directory
        fileName = d['fileName']     # names of files store in firestore i.e L-CC, R-CC , L-MLO , R-MLO

        currentTime = d['currentTime']  #currentTime is used as a result id
        date = d['date']

        pros_id = d["msg_id"]
        my_logger.info('processing msg_id : {}'.format(pros_id))
        
        try:
            parser = argparse.ArgumentParser(description='Run image-only model or image+heatmap model')
            parser.add_argument('--model-mode', default=MODELMODES.VIEW_SPLIT, type=str)
            parser.add_argument('--model-path', default='models/sample_image_model.p')
            parser.add_argument('--data-path', default='sample_output/{}/data.pkl'.format(pros_id))
            parser.add_argument('--image-path', default='sample_output/{}/cropped_images'.format(pros_id))
            parser.add_argument('--output-path', default='sample_output/{0}/image_predictions-{0}.csv'.format(pros_id))
            parser.add_argument('--batch-size', default=1, type=int)
            parser.add_argument('--seed', default=0, type=int)
            parser.add_argument('--use-heatmaps', action="store_true")
            parser.add_argument('--heatmaps-path')
            parser.add_argument('--use-augmentation', action="store_true")
            parser.add_argument('--use-hdf5', action="store_true")
            parser.add_argument('--num-epochs', default=1, type=int)
            parser.add_argument('--device-type', default="cpu", choices=['gpu', 'cpu'])
            parser.add_argument("--gpu-number", type=int, default=0)
            args = parser.parse_args()

            parameters = {
                "device_type": args.device_type,
                "gpu_number": args.gpu_number,
                "max_crop_noise": (100, 100),
                "max_crop_size_noise": 100,
                "image_path": args.image_path,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "augmentation": args.use_augmentation,
                "num_epochs": args.num_epochs,
                "use_heatmaps": args.use_heatmaps,
                "heatmaps_path": args.heatmaps_path,
                "use_hdf5": args.use_hdf5,
                "model_mode": args.model_mode,
                "model_path": args.model_path,
            }

            def load_model(parameters):
                """
                Loads trained cancer classifier
                """
                input_channels = 3 if parameters["use_heatmaps"] else 1
                model_class = {
                    MODELMODES.VIEW_SPLIT: models.SplitBreastModel,
                    MODELMODES.IMAGE: models.ImageBreastModel,
                }[parameters["model_mode"]]
                model = model_class(input_channels)
                model.load_state_dict(torch.load(parameters["model_path"])["model"])

                if (parameters["device_type"] == "gpu") and torch.has_cudnn:
                    device = torch.device("cuda:{}".format(parameters["gpu_number"]))
                else:
                    device = torch.device("cpu")
                model = model.to(device)
                model.eval()
                return model, device


            def run_model(model, device, exam_list, parameters):
                """
                Returns predictions of image only model or image+heatmaps model.
                Prediction for each exam is averaged for a given number of epochs.
                """
                random_number_generator = np.random.RandomState(parameters["seed"])

                image_extension = ".hdf5" if parameters["use_hdf5"] else ".png"

                with torch.no_grad():
                    predictions_ls = []
                    for datum in tqdm.tqdm(exam_list):
                        predictions_for_datum = []
                        loaded_image_dict = {view: [] for view in VIEWS.LIST}
                        loaded_heatmaps_dict = {view: [] for view in VIEWS.LIST}
                        for view in VIEWS.LIST:
                            for short_file_path in datum[view]:
                                loaded_image = loading.load_image(
                                    image_path=os.path.join(parameters["image_path"], short_file_path + image_extension),
                                    view=view,
                                    horizontal_flip=datum["horizontal_flip"],
                                )
                                if parameters["use_heatmaps"]:
                                    loaded_heatmaps = loading.load_heatmaps(
                                        benign_heatmap_path=os.path.join(parameters["heatmaps_path"], "heatmap_benign",
                                                                         short_file_path + ".hdf5"),
                                        malignant_heatmap_path=os.path.join(parameters["heatmaps_path"], "heatmap_malignant",
                                                                            short_file_path + ".hdf5"),
                                        view=view,
                                        horizontal_flip=datum["horizontal_flip"],
                                    )
                                else:
                                    loaded_heatmaps = None

                                loaded_image_dict[view].append(loaded_image)
                                loaded_heatmaps_dict[view].append(loaded_heatmaps)
                        for data_batch in tools.partition_batch(range(parameters["num_epochs"]), parameters["batch_size"]):
                            batch_dict = {view: [] for view in VIEWS.LIST}
                            for _ in data_batch:
                                for view in VIEWS.LIST:
                                    image_index = 0
                                    if parameters["augmentation"]:
                                        image_index = random_number_generator.randint(low=0, high=len(datum[view]))
                                    cropped_image, cropped_heatmaps = loading.augment_and_normalize_image(
                                        image=loaded_image_dict[view][image_index],
                                        auxiliary_image=loaded_heatmaps_dict[view][image_index],
                                        view=view,
                                        best_center=datum["best_center"][view][image_index],
                                        random_number_generator=random_number_generator,
                                        augmentation=parameters["augmentation"],
                                        max_crop_noise=parameters["max_crop_noise"],
                                        max_crop_size_noise=parameters["max_crop_size_noise"],
                                    )
                                    if loaded_heatmaps_dict[view][image_index] is None:
                                        batch_dict[view].append(cropped_image[:, :, np.newaxis])
                                    else:
                                        batch_dict[view].append(np.concatenate([
                                            cropped_image[:, :, np.newaxis],
                                            cropped_heatmaps,
                                        ], axis=2))

                            tensor_batch = {
                                view: torch.tensor(np.stack(batch_dict[view])).permute(0, 3, 1, 2).to(device)
                                for view in VIEWS.LIST
                            }
                            output = model(tensor_batch)
                            batch_predictions = compute_batch_predictions(output, mode=parameters["model_mode"])
                            pred_df = pd.DataFrame({k: v[:, 1] for k, v in batch_predictions.items()})
                            pred_df.columns.names = ["label", "view_angle"]
                            predictions = pred_df.T.reset_index().groupby("label").mean().T[LABELS.LIST].values
                            predictions_for_datum.append(predictions)
                        predictions_ls.append(np.mean(np.concatenate(predictions_for_datum, axis=0), axis=0))

                return np.array(predictions_ls)


            def compute_batch_predictions(y_hat, mode):
                """
                Format predictions from different heads
                """

                if mode == MODELMODES.VIEW_SPLIT:
                    assert y_hat[VIEWANGLES.CC].shape[1:] == (4, 2)
                    assert y_hat[VIEWANGLES.MLO].shape[1:] == (4, 2)
                    batch_prediction_tensor_dict = col.OrderedDict()
                    batch_prediction_tensor_dict[LABELS.LEFT_BENIGN, VIEWANGLES.CC] = y_hat[VIEWANGLES.CC][:, 0]
                    batch_prediction_tensor_dict[LABELS.LEFT_BENIGN, VIEWANGLES.MLO] = y_hat[VIEWANGLES.MLO][:, 0]
                    batch_prediction_tensor_dict[LABELS.RIGHT_BENIGN, VIEWANGLES.CC] = y_hat[VIEWANGLES.CC][:, 1]
                    batch_prediction_tensor_dict[LABELS.RIGHT_BENIGN, VIEWANGLES.MLO] = y_hat[VIEWANGLES.MLO][:, 1]
                    batch_prediction_tensor_dict[LABELS.LEFT_MALIGNANT, VIEWANGLES.CC] = y_hat[VIEWANGLES.CC][:, 2]
                    batch_prediction_tensor_dict[LABELS.LEFT_MALIGNANT, VIEWANGLES.MLO] = y_hat[VIEWANGLES.MLO][:, 2]
                    batch_prediction_tensor_dict[LABELS.RIGHT_MALIGNANT, VIEWANGLES.CC] = y_hat[VIEWANGLES.CC][:, 3]
                    batch_prediction_tensor_dict[LABELS.RIGHT_MALIGNANT, VIEWANGLES.MLO] = y_hat[VIEWANGLES.MLO][:, 3]
                    batch_prediction_dict = col.OrderedDict([
                        (k, np.exp(v.cpu().detach().numpy()))
                        for k, v in batch_prediction_tensor_dict.items()
                    ])
                elif mode == MODELMODES.IMAGE:
                    assert y_hat[VIEWS.L_CC].shape[1:] == (2, 2)
                    assert y_hat[VIEWS.R_CC].shape[1:] == (2, 2)
                    assert y_hat[VIEWS.L_MLO].shape[1:] == (2, 2)
                    assert y_hat[VIEWS.R_MLO].shape[1:] == (2, 2)
                    batch_prediction_tensor_dict = col.OrderedDict()
                    batch_prediction_tensor_dict[LABELS.LEFT_BENIGN, VIEWS.L_CC] = y_hat[VIEWS.L_CC][:, 0]
                    batch_prediction_tensor_dict[LABELS.LEFT_BENIGN, VIEWS.L_MLO] = y_hat[VIEWS.L_MLO][:, 0]
                    batch_prediction_tensor_dict[LABELS.RIGHT_BENIGN, VIEWS.R_CC] = y_hat[VIEWS.R_CC][:, 0]
                    batch_prediction_tensor_dict[LABELS.RIGHT_BENIGN, VIEWS.R_MLO] = y_hat[VIEWS.R_MLO][:, 0]
                    batch_prediction_tensor_dict[LABELS.LEFT_MALIGNANT, VIEWS.L_CC] = y_hat[VIEWS.L_CC][:, 1]
                    batch_prediction_tensor_dict[LABELS.LEFT_MALIGNANT, VIEWS.L_MLO] = y_hat[VIEWS.L_MLO][:, 1]
                    batch_prediction_tensor_dict[LABELS.RIGHT_MALIGNANT, VIEWS.R_CC] = y_hat[VIEWS.R_CC][:, 1]
                    batch_prediction_tensor_dict[LABELS.RIGHT_MALIGNANT, VIEWS.R_MLO] = y_hat[VIEWS.R_MLO][:, 1]

                    batch_prediction_dict = col.OrderedDict([
                        (k, np.exp(v.cpu().detach().numpy()))
                        for k, v in batch_prediction_tensor_dict.items()
                    ])
                else:
                    raise KeyError(mode)
                return batch_prediction_dict


            def load_run_save(data_path, output_path, parameters):
                """
                Outputs the predictions as csv file
                """
                exam_list = pickling.unpickle_from_file(data_path)
                model, device = load_model(parameters)
                predictions = run_model(model, device, exam_list, parameters)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                # Take the positive prediction
                df = pd.DataFrame(predictions, columns=LABELS.LIST)
                df.to_csv(output_path, index=False, float_format='%.4f')
            print('Stage 4a: Run Classifier (Image)')
            my_logger.info('Stage 4a: Run Classifier (Image)')
            load_run_save(
                data_path=args.data_path,
                output_path=args.output_path,
                parameters=parameters,
            )
            
        except Exception as e:
            print("ERROR:",e)
        
        credential_json_file = ['file_name']
        databaseURL = ['databaseURL']
        storageBucket = ['storageBucket']
        logs_bucket = ['logs_bucket']
        
        try:
            breast_cancer={}
            with open('sample_output/{0}/image_predictions-{0}.csv'.format(pros_id)) as f: #this code is reading image prediction csv result file of breast canceer classifier
                reader = csv.DictReader(f)           #and convert into json
                rows = list(reader)

            with open('image_predictions-{}.json'.format(pros_id),'w') as f:#writing json file for images prediction results
                json.dump(rows[0],f)

            with open('image_predictions-{}.json'.format(pros_id)) as json_file:
                breast_cancer = json.load(json_file)        #loading json data in dictionary
                breast_cancer.update({'error':'false'})
                result ={"Image_Predictions":breast_cancer}
            
        # might gives error in run-time for initialization to rule out such error checking initialization of firebase app again

            if (not len(firebase_admin._apps)):
                cred = credentials.Certificate(credential_json_file)
                fa=firebase_admin.initialize_app(cred, {"databaseURL": databaseURL,'storageBucket':storageBucket})
                fc=firebase_admin.firestore.client(fa)
                db = firestore.client()
                doc_ref = db.collection(u'stripe_customers/{0}/results'.format(userId)).document(currentTime)
                doc_ref.update(result)
                message.ack()
                my_logger.info('Breast cancer result saved (message acknowledged) :{0}'.format(breast_cancer))
            else:
                print('alredy initialize')
                db = firestore.client()
                doc_ref = db.collection(u'stripe_customers/{0}/results'.format(userId)).document(currentTime)
                doc_ref.update(result)
                message.ack()
                my_logger.info('Breast cancer result saved (message acknowledged) :{0}'.format(breast_cancer))
        except Exception as e:
            print(e)
            my_logger.info('Breast cancer model NOT processed ERROR : {}'.format(e))
        try:            
            storage_client = store.Client.from_service_account_json(credential_json_file)
            log_bucket = storage_client.get_bucket(logs_bucket)
            log_blob = log_bucket.blob('{0}/logs/{1}/{2}/stage-4a'.format(userId,date,currentTime))
            log_blob.upload_from_filename('{}.log'.format(msg_id))
            os.remove('./{}.log'.format(msg_id))
            os.remove('image_predictions-{}.json'.format(pros_id))
            shutil.rmtree("sample_data/{}".format(pros_id), ignore_errors=True)
            shutil.rmtree("sample_output/{}".format(pros_id), ignore_errors=True)
            print("waiting for new message...")

        except Exception as e:
            print("ERROR:",e)
def main():
    project_id = os.getenv("project_id") # from .env file here we'll check for development environment and production environment
    print('Project id : ',project_id)
    
    if project_id=="project_name":
        project_id = ['Project ID']
        subscription_name = ['subscription_name']
        print("processing in development environment")
        subscription_path = "projects/{0}/subscriptions/(1)".format(project_id,subscription_name)
    elif project_id=="project_name":
        project_id = ['Project ID']
        subscription_name = ['subscription_name']        
        print("processing in production environment")
        subscription_path = "projects/{0}/subscriptions/{1}".format(project_id,subscription_name)
    else:
        print("project_id not found")
    # Below three lines would trigger the processing of the message from uploads subscription
    handler = PubsubMessageHandler()
    subscriber = pubsub.SubscriberClient()
    future = subscriber.subscribe(subscription_path,handler.PubsubCallback) # this line will fetch new message from pubsub
    
    try:
        future.result()
        print("message processed",future.result())
    
    except KeyboardInterrupt:
        # User  exits the script early.
        future.cancel()             #to cancel the process with keyboard interrupt Ctrl+c
        print('Received keyboard interrupt, exiting...')
    
    print("inference-module terminated")        

if __name__ == '__main__':
    main()
