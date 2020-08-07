  
"""
The Originalfile belongs to breast density classifier under GNU Affero General Public License v3.0
and is available here: https://github.com/nyukat/breast_density_classifier.git
Neura Health made modifications to fils to run with Neuradetect application.
"""
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

import os
import sys
import json
import firebase_admin
from firebase_admin import firestore,storage,credentials
from google.cloud import pubsub
from google.cloud import storage as store
import logging
import logging.handlers
import png
import pydicom
import dotenv
dotenv.load_dotenv()
from PIL import Image
from resizeimage import resizeimage
from numpy import asarray
import shutil
from tensorflow.python.framework import ops

import argparse
import tensorflow as tf
import models_tf as models
import utils

class PubsubMessageHandler():

    def PubsubCallback(self,message):
        print("start pubsubcallback")
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

        url = d['url']               # url is using to copy images from firestore to root directory i.e neuradetect.py
        fileName = d['fileName']     # names of files store in firestore i.e L-CC, R-CC , L-MLO , R-MLO

        currentTime = d['currentTime']  #currentTime is used as a result id
        Time = d['time']
        date = d['date']
        
        credential_json_file = ['file_name']
        databaseURL = ['databaseURL']
        storageBucket = ['storageBucket']
        logs_bucket = ['logs_bucket']

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
                #belowcode will check image extention if it is .dcm
                else:
                    if url[x].endswith('.dcm'):
                        dcm_images = ['L-CC.dcm','R-CC.dcm','L-MLO.dcm','R-MLO.dcm']
                        blob = storage.bucket('dicom-poc.appspot.com').blob(url[x])
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
                        if data.shape > (2600, 2000):
                            with open(r'{0}/{1}'.format(msg_id,images[x]), 'r+b') as f:
                                    with Image.open(f) as image:
                                        cover = resizeimage.resize_cover(image, [2000,2600])
                                        cover.save(r'{0}/{1}'.format(msg_id,images[x]))

                            print("converted to png, Saving now..",png_filename)
                    else:
                        print('INVALID image format')
                    my_logger.info('images successfully saved')
        except BaseException as error:
            my_logger.error('{}'.format(error))
            my_logger.info('images does NOT saved')
            
        try:
            my_logger.info('Processing model now')
            parser = argparse.ArgumentParser(description='Run Inference')
            parser.add_argument('model_type')
            parser.add_argument('--bins-histogram', default=50)
            parser.add_argument('--model-path', default=None)
            parser.add_argument('--device-type', default="cpu")
            parser.add_argument('--gpu-number', default=0)
            parser.add_argument('--image-path', default="{}/".format(msg_id))
            args = parser.parse_args()

            parameters_ = {
                "model_type": args.model_type,
                "bins_histogram": args.bins_histogram,
                "model_path": args.model_path,
                "device_type": args.device_type,
                "image_path": args.image_path,
                "gpu_number": args.gpu_number,
                "input_size": (2600, 2000),
            }

            def optimistic_restore(session, save_file):
                reader = tf.train.NewCheckpointReader(save_file)
                saved_shapes = reader.get_variable_to_shape_map()
            var_names = sorted(
                [(var.name, var.name.split(':')[0]) for var in tf.global_variables() if var.name.split(':')[0] in saved_shapes])
            restore_vars = []

            with tf.variable_scope('', reuse=True):

                for var_name, saved_var_name in var_names:
                    curr_var = tf.get_variable(saved_var_name)
                    var_shape = curr_var.get_shape().as_list()

                    if var_shape == saved_shapes[saved_var_name]:
                        restore_vars.append(curr_var)

            saver = tf.train.Saver(restore_vars)
            saver.restore(session, save_file)
        except BaseException as error:
            my_logger.error('{}'.format(error))

        try:
            def inference(parameters, verbose=True):
                tf.set_random_seed(7)

                with tf.Graph().as_default():
                    with tf.device('/' + parameters['device_type']):
                        # initialize input holders
                        if parameters["model_type"] == 'cnn':
                            x_l_cc = tf.placeholder(tf.float32,
                                                    shape=[None, parameters['input_size'][0], parameters['input_size'][1], 1])
                            x_r_cc = tf.placeholder(tf.float32,
                                                    shape=[None, parameters['input_size'][0], parameters['input_size'][1], 1])
                            x_l_mlo = tf.placeholder(tf.float32,
                                                     shape=[None, parameters['input_size'][0], parameters['input_size'][1], 1])
                            x_r_mlo = tf.placeholder(tf.float32,
                                                     shape=[None, parameters['input_size'][0], parameters['input_size'][1], 1])
                            x = (x_l_cc, x_r_cc, x_l_mlo, x_r_mlo)
                            model_class = models.BaselineBreastModel
                        elif parameters["model_type"] == 'histogram':
                            x = tf.placeholder(tf.float32, shape=[None, parameters['bins_histogram'] * 4])
                            model_class = models.BaselineHistogramModel
                        else:
                            raise RuntimeError(parameters["model_type"])

                        # holders for dropout and Gaussian noise
                        nodropout_probability = tf.placeholder(tf.float32, shape=())
                        gaussian_noise_std = tf.placeholder(tf.float32, shape=())

                        # construct models
                        model = model_class(parameters, x, nodropout_probability, gaussian_noise_std)
                        y_prediction_density = model.y_prediction_density

                    # allocate computation resources
                    if parameters['device_type'] == 'gpu':
                        session_config = tf.ConfigProto()
                        session_config.gpu_options.visible_device_list = str(parameters['gpu_number'])
                    elif parameters['device_type'] == 'cpu':
                        session_config = tf.ConfigProto(device_count={'GPU': 0})
                    else:
                        raise RuntimeError(parameters['device_type'])

                    with tf.Session(config=session_config) as session:
                        session.run(tf.global_variables_initializer())

                        # loads the pre-trained parameters if it's provided
                        optimistic_restore(session, parameters['model_path'])

                        # load input images
                        datum_l_cc = utils.load_images(parameters['image_path'], 'L-CC')
                        datum_r_cc = utils.load_images(parameters['image_path'], 'R-CC')
                        datum_l_mlo = utils.load_images(parameters['image_path'], 'L-MLO')
                        datum_r_mlo = utils.load_images(parameters['image_path'], 'R-MLO')

                        # populate feed_dict for TF session
                        # No dropout and no gaussian noise in inference
                        feed_dict_by_model = {nodropout_probability: 1.0, gaussian_noise_std: 0.0}
                        if parameters["model_type"] == 'cnn':
                            feed_dict_by_model[x_l_cc] = datum_l_cc
                            feed_dict_by_model[x_r_cc] = datum_r_cc
                            feed_dict_by_model[x_l_mlo] = datum_l_mlo
                            feed_dict_by_model[x_r_mlo] = datum_r_mlo
                        elif parameters["model_type"] == 'histogram':
                            feed_dict_by_model[x] = utils.histogram_features_generator(
                                [datum_l_cc, datum_r_cc, datum_l_mlo, datum_r_mlo],
                                parameters,
                            )

                        # run the session for a prediction
                        prediction_density = session.run(y_prediction_density, feed_dict=feed_dict_by_model)

                        if verbose:
                            # nicely prints out the predictions
                            density0_prob = prediction_density[0, 0]
                            density1_prob = prediction_density[0, 1]
                            density2_prob = prediction_density[0, 2]
                            density3_prob = prediction_density[0, 3]
                            print('Density prediction:\n' +
                                  '\tAlmost entirely fatty (0):\t\t\t' + str(prediction_density[0, 0]) + '\n' +
                                  '\tScattered areas of fibroglandular density (1):\t' + str(prediction_density[0, 1]) + '\n' +
                                  '\tHeterogeneously dense (2):\t\t\t' + str(prediction_density[0, 2]) + '\n' +
                                  '\tExtremely dense (3):\t\t\t\t' + str(prediction_density[0, 3]) + '\n')

                            results={'Density_prediction':{
                            'Almost_entirely_fatty': str("%.2f"%(density0_prob*100)),
                            'Scattered_areas_of_fibroglandular_density': str("%.2f"%(density1_prob*100)),
                            'Heterogeneously_dense': str("%.2f"%(density2_prob*100)),
                            'Extremely_dense': str("%.2f"%(density3_prob*100))
                            }
                            }

                        try:
                            my_logger.info('Detection : {}'.format(results))

                            if (not len(firebase_admin._apps)):
                                cred = credentials.Certificate(credential_json_file)
                                fa=firebase_admin.initialize_app(cred, {"databaseURL": databaseURL,'storageBucket':storageBucket})
                                fc=firebase_admin.firestore.client(fa)
                                db = firestore.client()
                                doc_ref = db.collection(u'stripe_customers/{0}/results'.format(userId)).document(currentTime)
                                doc_ref.update(results)
                                my_logger.info('finish successfully and message acknowledged')
                                message.ack()
                            else:
                                print('alredy initialize')
                                db = firestore.client()
                                doc_ref = db.collection(u'stripe_customers/{0}/results'.format(userId)).document(currentTime)
                                doc_ref.update(results)
                                my_logger.info('finish successfully and message acknowledged')
                                message.ack()
                            
                            
                            storage_client = store.Client.from_service_account_json(credential_json_file)
                            log_bucket = storage_client.get_bucket(logs_bucket)
                            log_blob = log_bucket.blob('{0}/logs/{1}/{2}/density-log'.format(userId,date,currentTime))
                            log_blob.upload_from_filename('{}.log'.format(msg_id))
                            shutil.rmtree(msg_id)
                            os.remove('./{}.log'.format(msg_id))
                            return prediction_density[0]
                        except Exception as e :
                            my_logger.error('Detection : not done')

            if parameters_["model_path"] is None:
                if args.model_type == "histogram":
                    parameters_["model_path"] = "saved_models/BreastDensity_BaselineHistogramModel/model.ckpt"
                elif args.model_type == "cnn":
                    parameters_["model_path"] = "saved_models/BreastDensity_BaselineBreastModel/model.ckpt"
                else:
                    raise RuntimeError(parameters_['model_class'])

            inference(parameters_)
        
        except BaseException as error:
            images = ['L-CC.png','R-CC.png','L-MLO.png','R-MLO.png']
            for img in range(0,4):
                image = Image.open(r'{0}/{1}'.format(msg_id,images[img]))
                data = asarray(image)
            if data.shape < (2600,2000):
                invalid = {"Density_prediction":"invalid_image_size"}
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
            log_blob = log_bucket.blob('{0}/logs/{1}/{2}/density-log'.format(userId,date,currentTime))
            log_blob.upload_from_filename('{}.log'.format(msg_id))
            shutil.rmtree(msg_id)
            os.remove('./{}.log'.format(msg_id))
            message.ack()
            my_logger.error('{}'.format(error))
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
