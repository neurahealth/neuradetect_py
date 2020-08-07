"""
The Originalfile belongs to breast_cancer_classifier under GNU Affero General Public License v3.0
and is available here: https://github.com/nyukat/GMIC.git
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

import argparse
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
import tqdm
import cv2
import matplotlib.cm as cm
from src.utilities import pickling, tools
from src.modeling import gmic as gmic
from src.data_loading import loading
from src.constants import VIEWS, PERCENT_T_DICT

class PubsubMessageHandler():
    def PubsubCallback(self,message):
        msg_id =  message.message_id
        print(msg_id)

        filename = msg_id+'.log'
        my_logger = logging.getLogger(msg_id)
        my_logger.setLevel(logging.INFO)
        handler = logging.handlers.RotatingFileHandler(filename, maxBytes=20)
        my_logger.addHandler(handler)

        my_logger.info('Started with message_id : {}'.format(msg_id))

        sub_data = message.data.decode('utf-8')
        d = json.loads(sub_data)

        userId = d['userId']         # userid is used to create path to store results for same user
        print("User Id : ",userId)

        path = d['path']

        url = d['url']               # url is using to copy images from firestore to root directory i.e
        fileName = d['fileName']     # names of files store in firestore i.e L-CC, R-CC , L-MLO , R-MLO

        currentTime = d['currentTime']  #currentTime is used as a result id
        date = d['date']

        pros_id = d["msg_id"]
        my_logger.info('processing msg_id : {}'.format(pros_id))
        credential_json_file = ['file_name']
        databaseURL = ['databaseURL']
        storageBucket = ['storageBucket']
        os.mkdir('sample_output/{}'.format(pros_id))
        
        try:
            crop_img = ['L-CC.png','R-CC.png','L-MLO.png','R-MLO.png']
            for i in range(0,4):
                image = Image.open(r'sample_data/{0}/cropped_images/{1}'.format(pros_id,crop_img[i]))
                data = asarray(image)
                size = data.shape
                #print(data.shape)
                ht, wd  = size
                #print('width:  ', wd)
                #print('height: ', ht)
                if data.shape < (2944,1920):
                    my_logger.info("client image resolutions {0}-{1} is smaller than 1920,2944".format(crop_img[i],data.shape))
                    invalid = {"visualization":"invalid_image_size"}
                  
                    if (not len(firebase_admin._apps)):
                        cred = credentials.Certificate(credential_json_file)
                        fa=firebase_admin.initialize_app(cred, {"databaseURL": databaseURL,'storageBucket':storageBucket})
                        fc=firebase_admin.firestore.client(fa)
                        db = firestore.client()
                        doc_ref = db.collection(u'stripe_customers/{0}/results'.format(userId)).document(currentTime)
                        doc_ref.update(invalid)
                        message.ack()
                        my_logger.info('invalid image resolutions')
                    else:
                        print('alredy initialize')
                        db = firestore.client()
                        doc_ref = db.collection(u'stripe_customers/{0}/results'.format(userId)).document(currentTime)
                        doc_ref.update(invalid)
                        message.ack()
                        my_logger.info('invalid image resolutions')

                elif data.shape == (2944,1920):
                    my_logger.info('image resolution is correct')
                elif data.shape > (2944,1920):
                    #print("resizing")
                    with open(r'sample_data/{0}/cropped_images/{1}'.format(pros_id,crop_img[i]), 'r+b') as f:
                        with Image.open(f) as image:
                            cover = resizeimage.resize_cover(image, [1920,2944])
                            cover.save(r'sample_data/{0}/cropped_images/{1}'.format(pros_id,crop_img[i]))
                    my_logger.info('Oversize, Image resolution of {0} - Resize to>1920,2944'.format(crop_img[i]))
        except BaseException as error:
            print(error)
            my_logger.error('{}'.format(error))
            my_logger.info('images does NOT saved')

        try:
            my_logger.info('Processing model now')
            parser = argparse.ArgumentParser(description='Run GMIC on the sample data')
            parser.add_argument('--model-path', default='models/')
            parser.add_argument('--data-path', default='sample_data/{}/cropped_images/data.pkl'.format(pros_id))
            parser.add_argument('--image-path', default='sample_data/{}/cropped_images'.format(pros_id))
            parser.add_argument('--segmentation-path', default='sample_data/segmentation')
            parser.add_argument('--output-path', default='sample_output/{}'.format(pros_id))
            parser.add_argument('--device-type', default="cpu", choices=['gpu', 'cpu'])
            parser.add_argument("--gpu-number", type=int, default=0)
            parser.add_argument("--model-index", type=str, default="1")
            parser.add_argument("--visualization-flag", action="store_true", default=True)
            args = parser.parse_args()

            parameters = {
                "device_type": args.device_type,
                "gpu_number": args.gpu_number,
                "max_crop_noise": (100, 100),
                "max_crop_size_noise": 100,
                "image_path": args.image_path,
                "segmentation_path": args.segmentation_path,
                "output_path": args.output_path,
                # model related hyper-parameters
                "cam_size": (46,30),
                "K": 6,
                "crop_shape": (256, 256),
            }
            def visualize_example(input_img, saliency_maps, true_segs,patch_locations, patch_img, patch_attentions,save_dir, parameters):
                """
                Function that visualizes the saliency maps for an example
                """
                # colormap lists
                _, _, h, w = saliency_maps.shape
                _, _, H, W = input_img.shape

                # set up colormaps for benign and malignant
                alphas = np.abs(np.linspace(0, 0.95, 259))
                alpha_green = plt.cm.get_cmap('Greens')
                alpha_green._init()
                alpha_green._lut[:, -1] = alphas
                alpha_red = plt.cm.get_cmap('Reds')
                alpha_red._init()
                alpha_red._lut[:, -1] = alphas

                # create visualization template
                total_num_subplots = 4 + parameters["K"]
                figure = plt.figure(figsize=(30, 3))
                # input image + segmentation map
                subfigure = figure.add_subplot(1, total_num_subplots, 1)
                subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
                benign_seg, malignant_seg = true_segs
                if benign_seg is not None:
                    cm.Greens.set_under('w', alpha=0)
                    subfigure.imshow(benign_seg, alpha=0.85, cmap=cm.Greens, clim=[0.9, 1])
                if malignant_seg is not None:
                    cm.OrRd.set_under('w', alpha=0)
                    subfigure.imshow(malignant_seg, alpha=0.85, cmap=cm.OrRd, clim=[0.9, 1])
                subfigure.set_title("input image")
                subfigure.axis('off')

                # patch map
                subfigure = figure.add_subplot(1, total_num_subplots, 2)
                subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
                cm.YlGnBu.set_under('w', alpha=0)
                crop_mask = tools.get_crop_mask(
                    patch_locations[0, np.arange(parameters["K"]), :],
                    parameters["crop_shape"], (H, W),
                    "upper_left")
                subfigure.imshow(crop_mask, alpha=0.7, cmap=cm.YlGnBu, clim=[0.9, 1])
                if benign_seg is not None:
                    cm.Greens.set_under('w', alpha=0)
                    subfigure.imshow(benign_seg, alpha=0.85, cmap=cm.Greens, clim=[0.9, 1])
                if malignant_seg is not None:
                    cm.OrRd.set_under('w', alpha=0)
                    subfigure.imshow(malignant_seg, alpha=0.85, cmap=cm.OrRd, clim=[0.9, 1])
                subfigure.set_title("patch map")
                subfigure.axis('off')

                # class activation maps
                subfigure = figure.add_subplot(1, total_num_subplots, 4)
                subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
                resized_cam_malignant = cv2.resize(saliency_maps[0,1,:,:], (W, H))
                subfigure.imshow(resized_cam_malignant, cmap=alpha_red, clim=[0.0, 1.0])
                subfigure.set_title("SM: malignant")
                subfigure.axis('off')

                subfigure = figure.add_subplot(1, total_num_subplots, 3)
                subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
                resized_cam_benign = cv2.resize(saliency_maps[0,0,:,:], (W, H))
                subfigure.imshow(resized_cam_benign, cmap=alpha_green, clim=[0.0, 1.0])
                subfigure.set_title("SM: benign")
                subfigure.axis('off')


                # crops
                for crop_idx in range(parameters["K"]):
                    subfigure = figure.add_subplot(1, total_num_subplots, 5 + crop_idx)
                    subfigure.imshow(patch_img[0, crop_idx, :, :], cmap='gray', alpha=.8, interpolation='nearest',
                                     aspect='equal')
                    subfigure.axis('off')
                    # crops_attn can be None when we only need the left branch + visualization
                    subfigure.set_title("$\\alpha_{0} = ${1:.2f}".format(crop_idx, patch_attentions[crop_idx]))
                plt.savefig(save_dir, bbox_inches='tight', format="png", dpi=500)
                plt.close()


            def fetch_cancer_label_by_view(view, cancer_label):
                """
                Function that fetches cancer label using a view
                """
                if view in ["L-CC", "L-MLO"]:
                    return cancer_label["left_benign"], cancer_label["left_malignant"]
                elif view in ["R-CC", "R-MLO"]:
                    return cancer_label["right_benign"], cancer_label["right_malignant"]


            def run_model(model, exam_list, parameters, turn_on_visualization):
                """
                Run the model over images in sample_data.
                Save the predictions as csv and visualizations as png.
                """
                if (parameters["device_type"] == "gpu") and torch.has_cudnn:
                    device = torch.device("cuda:{}".format(parameters["gpu_number"]))
                else:
                    device = torch.device("cpu")
                model = model.to(device)
                model.eval()

                # initialize data holders
                pred_dict = {"image_index": [], "benign_pred": [], "malignant_pred": [],
                 "benign_label": 'no input', "malignant_label": 'no input'}
                with torch.no_grad():
                    # iterate through each exam
                    for datum in tqdm.tqdm(exam_list):
                        for view in VIEWS.LIST:
                            short_file_path = datum[view][0]
                            # load image
                            # the image is already flipped so no need to do it again
                            loaded_image = loading.load_image(
                                image_path=os.path.join(parameters["image_path"], short_file_path + ".png"),
                                view=view,
                                horizontal_flip=False,
                            )
                            loading.standard_normalize_single_image(loaded_image)
                            # load segmentation if available
                            benign_seg_path = os.path.join(parameters["segmentation_path"], "{0}_{1}".format(short_file_path, "benign.png"))
                            malignant_seg_path = os.path.join(parameters["segmentation_path"], "{0}_{1}".format(short_file_path, "malignant.png"))

                            benign_seg = np.zeros([1920,2944], dtype = int)
                            #benign_seg = None
                            malignant_seg = np.zeros([1920,2944], dtype = int)
                            #malignant_seg = None
                            if os.path.exists(benign_seg_path):
                                loaded_seg = loading.load_image(
                                    image_path=benign_seg_path,
                                    view=view,
                                    horizontal_flip=False,
                                )
                                benign_seg = loaded_seg
                            if os.path.exists(malignant_seg_path):
                                loaded_seg = loading.load_image(
                                    image_path=malignant_seg_path,
                                    view=view,
                                    horizontal_flip=False,
                                )
                                malignant_seg = loaded_seg
                            # convert python 2D array into 4D torch tensor in N,C,H,W format
                            loaded_image = np.expand_dims(np.expand_dims(loaded_image, 0), 0).copy()
                            tensor_batch = torch.Tensor(loaded_image).to(device)
                            # forward propagation
                            output = model(tensor_batch)
                            pred_numpy = output.data.cpu().numpy()
                            benign_pred, malignant_pred = pred_numpy[0, 0], pred_numpy[0, 1]
                            # save visualization
                            if turn_on_visualization:
                                saliency_maps = model.saliency_map.data.cpu().numpy()
                                patch_locations = model.patch_locations
                                patch_imgs = model.patches
                                patch_attentions = model.patch_attns[0, :].data.cpu().numpy()
                                save_dir = os.path.join(parameters["output_path"], "visualization", "{0}.png".format(short_file_path))
                                #b = np.zeros([2560 , 3328], dtype = int)
                                visualize_example(loaded_image, saliency_maps, [benign_seg, malignant_seg],
                                      patch_locations, patch_imgs, patch_attentions,
                                      save_dir, parameters)
                            # propagate holders
                            #benign_label, malignant_label = fetch_cancer_label_by_view(view, datum["cancer_label"])
                            pred_dict["image_index"].append(short_file_path)
                            pred_dict["benign_pred"].append(benign_pred)
                            pred_dict["malignant_pred"].append(malignant_pred)
                            #pred_dict["benign_label"].append(benign_label)
                            #pred_dict["malignant_label"].append(malignant_label)
                return pd.DataFrame(pred_dict)


            def run_single_model(model_path, data_path, parameters, turn_on_visualization):
                """
                Load a single model and run on sample data
                """
                # construct model
                model = gmic.GMIC(parameters)
                # load parameters
                if parameters["device_type"] == "gpu":
                    model.load_state_dict(torch.load(model_path), strict=False)
                else:
                    model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
                # load metadata
                exam_list = pickling.unpickle_from_file(data_path)
                # run the model on the dataset
                output_df = run_model(model, exam_list, parameters, turn_on_visualization)
                return output_df


            def start_experiment(model_path, data_path, output_path, model_index, parameters, turn_on_visualization):
                """
                Run the model on sample data and save the predictions as a csv file
                """
                # make sure model_index is valid
                valid_model_index = ["1", "2", "3", "4", "5", "ensemble"]
                assert model_index in valid_model_index, "Invalid model_index {0}. Valid options: {1}".format(model_index, valid_model_index)
                # create directories
                os.makedirs(output_path, exist_ok=True)
                os.makedirs(os.path.join(output_path, "visualization"), exist_ok=True)
                # do the average ensemble over predictions
                if model_index == "ensemble":
                    output_df_list = []
                    for i in range(1,6):
                        single_model_path = os.path.join(model_path, "sample_model_{0}.p".format(i))
                        # set percent_t for the model
                        parameters["percent_t"] = PERCENT_T_DICT[str(i)]
                        # only do visualization for the first model
                        need_visualization = i==1 and turn_on_visualization
                        current_model_output = run_single_model(single_model_path, data_path, parameters, need_visualization)
                        output_df_list.append(current_model_output)
                    all_prediction_df = pd.concat(output_df_list)
                    output_df = all_prediction_df.groupby("image_index").apply(lambda rows: pd.Series({"benign_pred":np.nanmean(rows["benign_pred"]),
                                  "malignant_pred": np.nanmean(rows["malignant_pred"]),
                                  "benign_label": rows.iloc[0]["benign_label"],
                                  "malignant_label": rows.iloc[0]["malignant_label"],
                                  })).reset_index()
                else:
                    # set percent_t for the model
                    parameters["percent_t"] = PERCENT_T_DICT[model_index]
                    single_model_path = os.path.join(model_path, "sample_model_{0}.p".format(model_index))
                    output_df = run_single_model(single_model_path, data_path, parameters, turn_on_visualization)

                # save the predictions
                output_df.to_csv(os.path.join(output_path, "predictions.csv"), index=False, float_format='%.4f')
            my_logger.info('Run Classifier visualization')
            print('Run Classifier')
            start_experiment(
                model_path=args.model_path,
                data_path=args.data_path,
                output_path=args.output_path,
                model_index=args.model_index,
                parameters=parameters,
                turn_on_visualization=args.visualization_flag,
                )

        except Exception as e:
            print(e)

        try:
            storage_client = store.Client.from_service_account_json(credential_json_file)
            log_bucket = storage_client.get_bucket(logs_bucket)

            visual_images = ['L-CC.png','R-CC.png','L-MLO.png','R-MLO.png']
            if (not len(firebase_admin._apps)):
                cred = credentials.Certificate(credential_json_file)
                fa=firebase_admin.initialize_app(cred, {"databaseURL": databaseURL,'storageBucket':storageBucket})
                fc=firebase_admin.firestore.client(fa)
                db = firestore.client()
                for i in range(0,4):
                    blob = storage.bucket(storageBucket).blob('{0}visualization/{1}'.format(path,visual_images[i]))
                    blob.upload_from_filename('sample_output/{0}/visualization/{1}'.format(pros_id,visual_images[i]))
                    message.ack()
                my_logger.info('visualizations saved successfully and message acknowledged')
            else:
                print('alredy initialize')
                db = firestore.client()
                for i in range(0,4):
                    blob = storage.bucket(storageBucket).blob('{0}visualization/{1}'.format(path,visual_images[i]))
                    blob.upload_from_filename('sample_output/{0}/visualization/{1}'.format(pros_id,visual_images[i]))
                    message.ack()
                my_logger.info('visualizations saved successfully and message acknowledged')
        
            log_blob = log_bucket.blob('{0}/logs/{1}/{2}/visualization'.format(userId,date,currentTime))
            log_blob.upload_from_filename('{}.log'.format(msg_id))
            #print(msg_id)
            os.remove('./{}.log'.format(msg_id))
            shutil.rmtree("sample_data/{}".format(pros_id), ignore_errors=True)
            shutil.rmtree("sample_output/{}".format(pros_id), ignore_errors=True)
        except Exception as e:
            my_logger.info('visualizations NOT saved ')
            log_blob = log_bucket.blob('{0}/logs/{1}/{2}/visualization'.format(userId,date,currentTime))
            log_blob.upload_from_filename('{}.log'.format(msg_id))
            os.remove('./{}.log'.format(msg_id))
            #print(e)

        print("waiting for new message...")

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
