"""
The Originalfile belongs to BIRADS_classifier under GNU Affero General Public License v3.0
and is available here: https://github.com/nyukat/BIRADS_classifier.git
Neura Health made modifications to run with neuradetect application.
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
import torch
import utils
import models_torch as models


class PubsubMessageHandler():
    def PubsubCallback(self,message):
        print("start pubsubcallback")
        #global message
        msg_id =  message.message_id
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

        bucket = d['bucket']         # bucket is used to complete path
        bkt = open('bucket.txt','w')
        bkt.write(bucket)
        bkt.close()

        path = d['path']

        url = d['url']               # url is using to copy images from firestore to root directory 
        fileName = d['fileName']     # names of files store in firestore i.e L-CC, R-CC , L-MLO , R-MLO

        currentTime = d['currentTime']  #currentTime is used as a result id
        date = d['date']

        credential_json_file = ['file_name']
        databaseURL = ['databaseURL']
        storageBucket = ['storageBucket']

        if (not len(firebase_admin._apps)):
            cred = credentials.Certificate(credential_json_file)
            fa=firebase_admin.initialize_app(cred, {"databaseURL": databaseURL,'storageBucket':storageBucket})
            fc=firebase_admin.firestore.client(fa)
            db = firestore.client()

        try:
            os.mkdir(msg_id)
            for x in range(0,4):
                images = ['L-CC.png','R-CC.png','L-MLO.png','R-MLO.png']
                if url[x].endswith('.png'):
                    blob = storage.bucket(storageBucket).blob(url[x])
                    blob.download_to_filename('{0}/{1}'.format(msg_id,images[x]))
                    image = Image.open(r'{0}/{1}'.format(msg_id,images[x]))
                    data = asarray(image)
                    if data.shape > (2600, 2000):
                        with open(r'{0}/{1}'.format(msg_id,images[x]), 'r+b') as f:
                                with Image.open(f) as image:
                                    cover = resizeimage.resize_cover(image, [2000,2600])
                                    cover.save(r'{0}/{1}'.format(msg_id,images[x]))
                else:
                    if url[x].endswith('.dcm'):
                        dcm_images = ['L-CC.dcm','R-CC.dcm','L-MLO.dcm','R-MLO.dcm']
                        blob = storage.bucket(storageBucket).blob(url[x])
                        blob.download_to_filename('{0}/{1}'.format(msg_id,dcm_images[x]))
                        dcm_images = ['L-CC.dcm','R-CC.dcm','L-MLO.dcm','R-MLO.dcm']
                        dicom_filename = (r'{0}/{1}'.format(msg_id,dcm_images[x]))
                        (".dcm file Detected, Converting dcm to png..",dcm_images[x])
                        png_filename = os.path.splitext(dcm_images[x])[0]
                        suffix = '.png'
                        png_filename = os.path.join(png_filename + suffix)
                        bitdepth=16
                        image = pydicom.read_file(dicom_filename).pixel_array
                        with open(r'{0}/{1}'.format(msg_id,images[x]), 'wb') as f:
                            writer = png.Writer(height=image.shape[0], width=image.shape[1], bitdepth=bitdepth, greyscale=True)
                            writer.write(f, image.tolist())

                        image = Image.open(r'{0}/{1}'.format(msg_id,images[x]))
                        data = asarray(image)
                        if data.shape >= (2600, 2000):
                            with open(r'{0}/{1}'.format(msg_id,images[x]), 'r+b') as f:
                                    with Image.open(f) as image:
                                        cover = resizeimage.resize_cover(image, [2000,2600])
                                        cover.save(r'{0}/{1}'.format(msg_id,images[x]))
                            print("converted to png, Saving now..",png_filename)
                            my_logger.info('images successfully saved')
                        else:
                            print('INVALID image size')

        except BaseException as error:
            my_logger.error('{}'.format(error))
            my_logger.info('images does NOT saved')

        try:
            my_logger.info('Processing model now')
            parser = argparse.ArgumentParser(description='Run Inference')
            parser.add_argument('--model-path', default='saved_models/model.p')
            parser.add_argument('--device-type', default="cpu")
            parser.add_argument('--gpu-number', default=0, type=int)
            parser.add_argument('--image-path', default="{}/".format(msg_id))
            args = parser.parse_args()

            parameters_ = {
                "model_path": args.model_path,
                "device_type": args.device_type,
                "gpu_number": args.gpu_number,
                "image_path": args.image_path,
                "input_size": (2600, 2000),
                }
                                    
            def inference(parameters, verbose=True):
                """
                Function that creates a model, loads the parameters, and makes a prediction
                :param parameters: dictionary of parameters
                :param verbose: Whether to print predicted probabilities
                :return: Predicted probabilities for each class
                """
                # resolve device
                device = torch.device(
                    "cuda:{}".format(parameters["gpu_number"]) if parameters["device_type"] == "gpu"
                    else "cpu"
                )

                # construct models
                model = models.BaselineBreastModel(device, nodropout_probability=1.0, gaussian_noise_std=0.0).to(device)
                model.load_state_dict(torch.load(parameters["model_path"]))

                # load input images and prepare data
                datum_l_cc = utils.load_images(parameters['image_path'], 'L-CC')
                datum_r_cc = utils.load_images(parameters['image_path'], 'R-CC')
                datum_l_mlo = utils.load_images(parameters['image_path'], 'L-MLO')
                datum_r_mlo = utils.load_images(parameters['image_path'], 'R-MLO')
                x = {
                    "L-CC": torch.Tensor(datum_l_cc).permute(0, 3, 1, 2).to(device),
                    "L-MLO": torch.Tensor(datum_l_mlo).permute(0, 3, 1, 2).to(device),
                    "R-CC": torch.Tensor(datum_r_cc).permute(0, 3, 1, 2).to(device),
                    "R-MLO": torch.Tensor(datum_r_mlo).permute(0, 3, 1, 2).to(device),
                }

                # run prediction
                with torch.no_grad():
                    prediction_birads = model(x).cpu().numpy()

                if verbose:
                    # nicely prints out the predictions
                    birads0_prob = prediction_birads[0][0]
                    birads1_prob = prediction_birads[0][1]
                    birads2_prob = prediction_birads[0][2]
                    print('BI-RADS prediction:\n' +
                          '\tBI-RADS 0:\t' + str(birads0_prob) + '\n' +
                          '\tBI-RADS 1:\t' + str(birads1_prob) + '\n' +
                          '\tBI-RADS 2:\t' + str(birads2_prob))

                results={'BI_RADS_prediction':{
                'BI_RADS_zero': str("%.2f"%(birads0_prob*100)),
                'BI_RADS_one': str("%.2f"%(birads1_prob*100)),
                'BI_RADS_two': str("%.2f"%(birads2_prob*100))
                }
                }

                credential_json_file = ['file_name']
                databaseURL = ['databaseURL']
                storageBucket = ['storageBucket']
                logs_bucket = ['logs_bucket']
                              
                try:
                    my_logger.info('Detection : {}'.format(results))

                    if (not len(firebase_admin._apps)):
                        cred = credentials.Certificate(credential_json_file)
                        fa=firebase_admin.initialize_app(cred, {"databaseURL": databaseURL,'storageBucket':storageBucket})
                        fc=firebase_admin.firestore.client(fa)
                        db = firestore.client()
                        doc_ref = db.collection(u'stripe_customers/{0}/results'.format(userId)).document(currentTime)
                        doc_ref.update(results)
                        message.ack()
                        my_logger.info('finish successfully and message acknowledge')
                    else:
                        print('alredy initialize')
                        db = firestore.client()
                        doc_ref = db.collection(u'stripe_customers/{0}/results'.format(userId)).document(currentTime)
                        doc_ref.update(results)
                        message.ack()
                        my_logger.info('finish successfully and message acknowledge')

                except Exception as e :
                    print(e)
                    my_logger.error(e,'results NOT save to firebase-database')

                storage_client = store.Client.from_service_account_json(credential_json_file)
                log_bucket = storage_client.get_bucket(logs_bucket)
                log_blob = log_bucket.blob('{0}/logs/{1}/{2}/BIRADS-logs'.format(userId,date,currentTime))
                log_blob.upload_from_filename('{}.log'.format(msg_id))
                shutil.rmtree(msg_id)
                os.remove('./{}.log'.format(msg_id))
                return prediction_birads[0]

            inference(parameters_)
        except Exception as e:
            images = ['L-CC.png','R-CC.png','L-MLO.png','R-MLO.png']
            for img in range(0,4):
                image = Image.open(r'{0}/{1}'.format(msg_id,images[img]))
                data = asarray(image)
            if data.shape < (2600,2000):
                invalid = {"BI_RADS_prediction":"invalid_image_size"}
                if (not len(firebase_admin._apps)):
                    cred = credentials.Certificate(credential_json_file)
                    fa=firebase_admin.initialize_app(cred, {"databaseURL": databaseURL,'storageBucket':storageBucket})
                    fc=firebase_admin.firestore.client(fa)
                    db = firestore.client()
                    doc_ref = db.collection(u'stripe_customers/{0}/results'.format(userId)).document(currentTime)
                    doc_ref.update(invalid)
                    my_logger.info('invalid image resolutions')
                else:
                    print('alredy initialize')
                    db = firestore.client()
                    doc_ref = db.collection(u'stripe_customers/{0}/results'.format(userId)).document(currentTime)
                    doc_ref.update(invalid)
                    my_logger.info('invalid image resolutions')
            
            storage_client = store.Client.from_service_account_json(credential_json_file)
            log_bucket = storage_client.get_bucket(logs_bucket)
            log_blob = log_bucket.blob('{0}/logs/{1}/{2}/BIRADS-logs'.format(userId,date,currentTime))
            log_blob.upload_from_filename('{}.log'.format(msg_id))
            shutil.rmtree(msg_id)
            os.remove('./{}.log'.format(msg_id))
            message.ack()
            my_logger.error(e,'Detection : not done')
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
