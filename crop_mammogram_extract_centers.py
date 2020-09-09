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

##model imports
import os
from multiprocessing import Pool
import argparse
from functools import partial
import scipy.ndimage
import numpy as np
import pandas as pd

import src.utilities.pickling as pickling
import src.utilities.reading_images as reading_images
import src.utilities.saving_images as saving_images
import src.utilities.data_handling as data_handling

#optimal centers imports`
import argparse
import numpy as np
import os
from itertools import repeat
from multiprocessing import Pool

from src.constants import INPUT_SIZE_DICT
import src.utilities.pickling as pickling
import src.utilities.data_handling as data_handling
import src.utilities.reading_images as reading_images
import src.data_loading.loading as loading
import src.optimal_centers.calc_optimal_centers as calc_optimal_centers

#to do
credential_json_file = ['file_name']
databaseURL = ['databaseURL']
storageBucket = ['storageBucket']
logs_bucket = ['logs_bucket']

class PubsubMessageHandler():
    def PubsubCallback(self,message):
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
#        print("User Id : ",userId)

        bucket = d['bucket']         # bucket is used to complete path

        path = d['path']

        url = d['url']               # url is using to copy images from firestore to root directory
        fileName = d['fileName']     # names of files store in firestore i.e L-CC, R-CC , L-MLO , R-MLO

        currentTime = d['currentTime']  #currentTime is used as a result id
        date = d['date']

        msg_info= {"msg_id":"{}".format(msg_id),"userId":"{}".format(userId),"bucket":"{}".format(bucket),"path":"{}".format(path),"url":"{}".format(url),"fileName":"{}".format(fileName),"currentTime":"{}".format(currentTime),"date":"{}".format(date)}
        
        #download images from storage
        if (not len(firebase_admin._apps)):
             cred = credentials.Certificate(credential_json_file)
             fa=firebase_admin.initialize_app(cred, {"databaseURL": databaseURL,'storageBucket':storageBucket})
             fc=firebase_admin.firestore.client(fa)
             db = firestore.client()

        try:
            dir_path = 'sample_data/{}'.format(msg_id)
            os.mkdir(dir_path)

            for x in range(0,4):
                images = ['L-CC.png','R-CC.png','L-MLO.png','R-MLO.png']
                if url[x].endswith('.png'):
                    blob = storage.bucket(storageBucket).blob(url[x])
                    blob.download_to_filename('sample_data/{0}/{1}'.format(msg_id,images[x]))

                #images are dcm extension
                else:
                    if url[x].endswith('.dcm'):
                        dcm_images = ['L-CC.dcm','R-CC.dcm','L-MLO.dcm','R-MLO.dcm']
                        blob = storage.bucket('dicom-poc.appspot.com').blob(url[x])
                        blob.download_to_filename('sample_data/{0}/{1}'.format(msg_id,dcm_images[x]))
                        #dcm_images = ['L-CC.dcm','R-CC.dcm','L-MLO.dcm','R-MLO.dcm']
                        dicom_filename = (r'sample_data/{0}/{1}'.format(msg_id,dcm_images[x]))
                        (".dcm file Detected, Converting dcm to png..",dcm_images[x])
                        png_filename = os.path.splitext(dcm_images[x])[0]
                        suffix = '.png'
                        png_filename = os.path.join(png_filename + suffix)
                        bitdepth=16
                        image = pydicom.read_file(dicom_filename).pixel_array
                        with open(r'sample_data/{0}/{1}'.format(msg_id,images[x]), 'wb') as f:
                            writer = png.Writer(height=image.shape[0], width=image.shape[1], bitdepth=bitdepth, greyscale=True)
                            writer.write(f, image.tolist())
                        os.remove(r'sample_data/{0}/{1}'.format(msg_id,dcm_images[x]))

        except BaseException as error:
            my_logger.error('{}'.format(error))
            my_logger.info('images does NOT saved')
        
        #check over size images
        try:
            image_C=['L-CC.png','R-CC.png','L-MLO.png','R-MLO.png']
            for img in range(0,4):
                image = Image.open(r'sample_data/{0}/{1}'.format(msg_id,image_C[img]))
                data = asarray(image)
                if data.shape >= (4084,3328):
                    with open(r'sample_data/{0}/{1}'.format(msg_id,image_C[img]), 'r+b') as f:
                        with Image.open(f) as image:
                            cover = resizeimage.resize_cover(image, [3328,4084])
                            cover.save(r'sample_data/{0}/{1}'.format(msg_id,image_C[img]))

                    my_logger.info('resizing {0} image resolutions'.format(image_C[img]))

        except BaseException as error:
            my_logger.error('{}'.format(error))
            print(error)
            my_logger.info('image resolutions does not matched')

        try:
            parser = argparse.ArgumentParser(description='Remove background of image and save cropped files')
            parser.add_argument('--input-data-folder', default='sample_data/{}'.format(msg_id))
            parser.add_argument('--output-data-folder', default='sample_output/{}/cropped_images'.format(msg_id))
            parser.add_argument('--exam-list-path',default='sample_data/exam_list_before_cropping_smaller.pkl')
            parser.add_argument('--cropped-exam-list-path', default='sample_output/{}/cropped_images/cropped_exam_list.pkl'.format(msg_id))
            parser.add_argument('--num-processes', default=1, type=int)
            parser.add_argument('--num-iterations', default=100, type=int)
            parser.add_argument('--buffer-size', default=50, type=int)
            args = parser.parse_args()

            def get_masks_and_sizes_of_connected_components(img_mask):
                """
                Finds the connected components from the mask of the image
                """
                mask, num_labels = scipy.ndimage.label(img_mask)

                mask_pixels_dict = {}
                for i in range(num_labels+1):
                    this_mask = (mask == i)
                    if img_mask[this_mask][0] != 0:
                        # Exclude the 0-valued mask
                        mask_pixels_dict[i] = np.sum(this_mask)

                return mask, mask_pixels_dict


            def get_mask_of_largest_connected_component(img_mask):
                """
                Finds the largest connected component from the mask of the image
                """
                mask, mask_pixels_dict = get_masks_and_sizes_of_connected_components(img_mask)
                largest_mask_index = pd.Series(mask_pixels_dict).idxmax()
                largest_mask = mask == largest_mask_index
                return largest_mask


            def get_edge_values(img, largest_mask, axis):
                """
                Finds the bounding box for the largest connected component
                """
                assert axis in ["x", "y"]
                has_value = np.any(largest_mask, axis=int(axis == "y"))
                edge_start = np.arange(img.shape[int(axis == "x")])[has_value][0]
                edge_end = np.arange(img.shape[int(axis == "x")])[has_value][-1] + 1
                return edge_start, edge_end

            def get_bottommost_pixels(img, largest_mask, y_edge_bottom):
                """
                Gets the bottommost nonzero pixels of dilated mask before cropping.
                """
                bottommost_nonzero_y = y_edge_bottom - 1
                bottommost_nonzero_x = np.arange(img.shape[1])[largest_mask[bottommost_nonzero_y, :] > 0]
                return bottommost_nonzero_y, bottommost_nonzero_x


            def get_distance_from_starting_side(img, mode, x_edge_left, x_edge_right):

                """
                If we fail to recover the original shape as a result of erosion-dilation
                on the side where the breast starts to appear in the image,
                we record this information.
                """
                if mode == "left":
                    return img.shape[1] - x_edge_right
                else:
                    return x_edge_left


            def include_buffer_y_axis(img, y_edge_top, y_edge_bottom, buffer_size):

                """
                Includes buffer in all sides of the image in y-direction
                """
                if y_edge_top > 0:
                    y_edge_top -= min(y_edge_top, buffer_size)
                if y_edge_bottom < img.shape[0]:
                    y_edge_bottom += min(img.shape[0] - y_edge_bottom, buffer_size)
                return y_edge_top, y_edge_bottom


            def include_buffer_x_axis(img, mode, x_edge_left, x_edge_right, buffer_size):

                """
                Includes buffer in only one side of the image in x-direction
                """
                if mode == "left":
                    if x_edge_left > 0:
                        x_edge_left -= min(x_edge_left, buffer_size)
                else:
                    if x_edge_right < img.shape[1]:
                        x_edge_right += min(img.shape[1] - x_edge_right, buffer_size)
                return x_edge_left, x_edge_right


            def convert_bottommost_pixels_wrt_cropped_image(mode, bottommost_nonzero_y, bottommost_nonzero_x,
                                                            y_edge_top, x_edge_right, x_edge_left):

                """
                Once the image is cropped, adjusts the bottommost pixel values which was originally w.r.t. the original image
                """
                bottommost_nonzero_y -= y_edge_top
                if mode == "left":
                    bottommost_nonzero_x = x_edge_right - bottommost_nonzero_x  # in this case, not in sorted order anymore.
                    bottommost_nonzero_x = np.flip(bottommost_nonzero_x, 0)
                else:
                    bottommost_nonzero_x -= x_edge_left
                return bottommost_nonzero_y, bottommost_nonzero_x


            def get_rightmost_pixels_wrt_cropped_image(mode, largest_mask_cropped, find_rightmost_from_ratio):

                """
                Ignores top find_rightmost_from_ratio of the image and searches the rightmost nonzero pixels
                of the dilated mask from the bottom portion of the image.
                """
                ignore_height = int(largest_mask_cropped.shape[0] * find_rightmost_from_ratio)
                rightmost_pixel_search_area = largest_mask_cropped[ignore_height:, :]
                rightmost_pixel_search_area_has_value = np.any(rightmost_pixel_search_area, axis=0)
                rightmost_nonzero_x = np.arange(rightmost_pixel_search_area.shape[1])[
                    rightmost_pixel_search_area_has_value][-1 if mode == 'right' else 0]
                rightmost_nonzero_y = np.arange(rightmost_pixel_search_area.shape[0])[
                    rightmost_pixel_search_area[:, rightmost_nonzero_x] > 0] + ignore_height

                # rightmost pixels are already found w.r.t. newly cropped image, except that we still need to
                #   reflect horizontal_flip
                if mode == "left":
                    rightmost_nonzero_x = largest_mask_cropped.shape[1] - rightmost_nonzero_x

                return rightmost_nonzero_y, rightmost_nonzero_x


            def crop_img_from_largest_connected(img, mode, erode_dialate=True, iterations=100,
                                                buffer_size=50, find_rightmost_from_ratio=1/3):

                assert mode in ("left", "right")

                img_mask = img > 0

                # Erosion in order to remove thin lines in the background
                if erode_dialate:
                    img_mask = scipy.ndimage.morphology.binary_erosion(img_mask, iterations=iterations)

                # Select mask for largest connected component
                largest_mask = get_mask_of_largest_connected_component(img_mask)

                # Dilation to recover the original mask, excluding the thin lines
                if erode_dialate:
                    largest_mask = scipy.ndimage.morphology.binary_dilation(largest_mask, iterations=iterations)

                # figure out where to crop
                y_edge_top, y_edge_bottom = get_edge_values(img, largest_mask, "y")
                x_edge_left, x_edge_right = get_edge_values(img, largest_mask, "x")

                # extract bottommost pixel info
                bottommost_nonzero_y, bottommost_nonzero_x = get_bottommost_pixels(img, largest_mask, y_edge_bottom)

                # include maximum 'buffer_size' more pixels on both sides just to make sure we don't miss anything
                y_edge_top, y_edge_bottom = include_buffer_y_axis(img, y_edge_top, y_edge_bottom, buffer_size)

                # If cropped image not starting from corresponding edge, they are wrong. Record the distance, will reject if not 0.
                distance_from_starting_side = get_distance_from_starting_side(img, mode, x_edge_left, x_edge_right)

                # include more pixels on either side just to make sure we don't miss anything, if the next column
                # contains non-zero value and isn't noise
                x_edge_left, x_edge_right = include_buffer_x_axis(img, mode, x_edge_left, x_edge_right, buffer_size)

                # convert bottommost pixel locations w.r.t. newly cropped image. Flip if necessary.
                bottommost_nonzero_y, bottommost_nonzero_x = convert_bottommost_pixels_wrt_cropped_image(
                    mode,
                    bottommost_nonzero_y,
                    bottommost_nonzero_x,
                    y_edge_top,
                    x_edge_right,
                    x_edge_left
                )

                # calculate rightmost point from bottom portion of the image w.r.t. cropped image. Flip if necessary.
                rightmost_nonzero_y, rightmost_nonzero_x = get_rightmost_pixels_wrt_cropped_image(
                    mode,
                    largest_mask[y_edge_top: y_edge_bottom, x_edge_left: x_edge_right],
                    find_rightmost_from_ratio
                )

                # save window location in medical mode, but everything else in training mode
                return (y_edge_top, y_edge_bottom, x_edge_left, x_edge_right), \
                    ((rightmost_nonzero_y[0], rightmost_nonzero_y[-1]), rightmost_nonzero_x), \
                    (bottommost_nonzero_y, (bottommost_nonzero_x[0], bottommost_nonzero_x[-1])), \
                    distance_from_starting_side


            def image_orientation(horizontal_flip, side):

                """
                Returns the direction where the breast should be facing in the original image
                This information is used in cropping.crop_img_horizontally_from_largest_connected
                """
                assert horizontal_flip in ['YES', 'NO'], "Wrong horizontal flip"
                assert side in ['L', 'R'], "Wrong side"
                if horizontal_flip == 'YES':
                    if side == 'R':
                        return 'right'
                    else:
                        return 'left'
                else:
                    if side == 'R':
                        return 'left'
                    else:
                        return 'right'

            def crop_mammogram(input_data_folder, exam_list_path, cropped_exam_list_path, output_data_folder,
                               num_processes, num_iterations, buffer_size):

                """
                In parallel, crops mammograms in DICOM format found in input_data_folder and save as png format in
                output_data_folder and saves new image list in cropped_image_list_path
                """
                exam_list = pickling.unpickle_from_file(exam_list_path)

                image_list = data_handling.unpack_exam_into_images(exam_list)

                if os.path.exists(output_data_folder):
                    # Prevent overwriting to an existing directory
                    print("Error: the directory to save cropped images already exists.")
                    return
                else:
                    os.makedirs(output_data_folder)
                #global crop_mammogram_one_image_func
                crop_mammogram_one_image_func = partial(
                    crop_mammogram_one_image_short_path,
                    input_data_folder=input_data_folder,
                    output_data_folder=output_data_folder,
                    num_iterations=num_iterations,
                    buffer_size=buffer_size,
                )
                with Pool(num_processes) as pool:
                    cropped_image_info = pool.map(crop_mammogram_one_image_func, image_list)

                window_location_dict = dict([x[0] for x in cropped_image_info])
                rightmost_points_dict = dict([x[1] for x in cropped_image_info])
                bottommost_points_dict = dict([x[2] for x in cropped_image_info])
                distance_from_starting_side_dict = dict([x[3] for x in cropped_image_info])

                data_handling.add_metadata(exam_list, "window_location", window_location_dict)
                data_handling.add_metadata(exam_list, "rightmost_points", rightmost_points_dict)
                data_handling.add_metadata(exam_list, "bottommost_points", bottommost_points_dict)
                data_handling.add_metadata(exam_list, "distance_from_starting_side", distance_from_starting_side_dict)


                pickling.pickle_to_file(cropped_exam_list_path, exam_list)


            def crop_mammogram_one_image(scan, input_file_path, output_file_path, num_iterations, buffer_size):
                """
                Crops a mammogram, saves as png file, includes the following additional information:
                    - window_location: location of cropping window w.r.t. original dicom image so that segmentation
                       map can be cropped in the same way for training.
                    - rightmost_points: rightmost nonzero pixels after correctly being flipped
                    - bottommost_points: bottommost nonzero pixels after correctly being flipped
                    - distance_from_starting_side: number of zero columns between the start of the image and start of
                       the largest connected component w.r.t. original dicom image.
                """

                image = reading_images.read_image_png(input_file_path)
                try:
                    # error detection using erosion. Also get cropping information for this image.
                    cropping_info = crop_img_from_largest_connected(
                        image,
                        image_orientation(scan['horizontal_flip'], scan['side']),
                        True,
                        num_iterations,
                        buffer_size,
                        1/3
                    )
                except Exception as error:
                    print(input_file_path, "\n\tFailed to crop image because image is invalid.", str(error))
                else:

                    top, bottom, left, right = cropping_info[0]

                    target_parent_dir = os.path.split(output_file_path)[0]
                    if not os.path.exists(target_parent_dir):
                        os.makedirs(target_parent_dir)

                    try:
                        saving_images.save_image_as_png(image[top:bottom, left:right], output_file_path)
                    except Exception as error:
                        print(input_file_path, "\n\tError while saving image.", str(error))

                    return cropping_info
            global crop_mammogram_one_image_short_path
            def crop_mammogram_one_image_short_path(scan, input_data_folder, output_data_folder,
                                                    num_iterations, buffer_size):


                """
                Crops a mammogram from a short_file_path
                See: crop_mammogram_one_image
                """
                full_input_file_path = os.path.join(input_data_folder, scan['short_file_path']+'.png')
                full_output_file_path = os.path.join(output_data_folder, scan['short_file_path'] + '.png')
                cropping_info = crop_mammogram_one_image(
                    scan=scan,
                    input_file_path=full_input_file_path,
                    output_file_path=full_output_file_path,
                    num_iterations=num_iterations,
                    buffer_size=buffer_size,
                )
                return list(zip([scan['short_file_path']] * 4, cropping_info))

            def extract_center(datum, image):
                """
                Compute the optimal center for an image
                """
                image = loading.flip_image(image, datum["full_view"], datum['horizontal_flip'])
                if datum["view"] == "MLO":
                    tl_br_constraint = calc_optimal_centers.get_bottomrightmost_pixel_constraint(
                        rightmost_x=datum["rightmost_points"][1],
                        bottommost_y=datum["bottommost_points"][0],
                    )
                elif datum["view"] == "CC":
                    tl_br_constraint = calc_optimal_centers.get_rightmost_pixel_constraint(
                        rightmost_x=datum["rightmost_points"][1]
                    )
                else:
                    raise RuntimeError(datum["view"])
                optimal_center = calc_optimal_centers.get_image_optimal_window_info(
                    image,
                    com=np.array(image.shape) // 2,
                    window_dim=np.array(INPUT_SIZE_DICT[datum["full_view"]]),
                    tl_br_constraint=tl_br_constraint,
                )
                return optimal_center["best_center_y"], optimal_center["best_center_x"]

            global load_and_extract_center
            def load_and_extract_center(datum, data_prefix):
                """
                Load image and computer optimal center
                """
                full_image_path = os.path.join(data_prefix, datum["short_file_path"] + '.png')
                image = reading_images.read_image_png(full_image_path)
                return datum["short_file_path"], extract_center(datum, image)


            def get_optimal_centers(data_list, data_prefix, num_processes=1):
                """
                Compute optimal centers for each image in data list
                """
                global pool
                pool = Pool(num_processes)
                result = pool.starmap(load_and_extract_center, zip(data_list, repeat(data_prefix)))
                return dict(result)


            def main(cropped_exam_list_path, data_prefix, output_exam_list_path, num_processes=1):
                exam_list = pickling.unpickle_from_file(cropped_exam_list_path)
                data_list = data_handling.unpack_exam_into_images(exam_list, cropped=True)
                optimal_centers = get_optimal_centers(
                    data_list=data_list,
                    data_prefix=data_prefix,
                    num_processes=num_processes
                )
                data_handling.add_metadata(exam_list, "best_center", optimal_centers)
                os.makedirs(os.path.dirname(output_exam_list_path), exist_ok=True)
                pickling.pickle_to_file(output_exam_list_path, exam_list)

            print('Stage 1: Crop Mammograms')
            my_logger.info("processing Stage 1: Crop Mammograms")
            crop_mammogram(
                input_data_folder=args.input_data_folder,
                exam_list_path=args.exam_list_path,
                cropped_exam_list_path=args.cropped_exam_list_path,
                output_data_folder=args.output_data_folder,
                num_processes=args.num_processes,
                num_iterations=args.num_iterations,
                buffer_size=args.buffer_size,
                )
            
            parser = argparse.ArgumentParser(description='Compute and Extract Optimal Centers')
            parser.add_argument('--cropped-exam-list-path',default = "sample_output/{}/cropped_images/cropped_exam_list.pkl".format(msg_id))
            parser.add_argument('--data-prefix',default = 'sample_output/{}/cropped_images'.format(msg_id))
            parser.add_argument('--output-exam-list-path', default= 'sample_output/{}/data.pkl'.format(msg_id))
            parser.add_argument('--num-processes', default=1)
            args1 = parser.parse_args()

            print('Stage 2: Extract Centers')
            my_logger.info("processing Stage 2: Extract centers")
            main(
                cropped_exam_list_path=args1.cropped_exam_list_path,
                data_prefix=args1.data_prefix,
                output_exam_list_path=args1.output_exam_list_path,
                num_processes=int(args1.num_processes)
                )

        except Exception as e:
            print(e)
            my_logger.info('ERROR: {}'.format(e))

        try:
            move_img = ['L-CC.png','R-CC.png','L-MLO.png','R-MLO.png']
            os.mkdir("GMIC/sample_data/{}".format(msg_id))
            os.mkdir('GMIC/sample_data/{}/cropped_images'.format(msg_id))

            for i in range(0,4):
                source = 'sample_output/{0}/cropped_images/{1}'.format(msg_id,move_img[i])
                destination = 'GMIC/sample_data/{0}/cropped_images/{1}'.format(msg_id,move_img[i])
                shutil.copyfile(source, destination)
            source1 = 'sample_output/{}/data.pkl'.format(msg_id)
            destination1 = ('GMIC/sample_data/{}/cropped_images/data.pkl'.format(msg_id))
            shutil.copyfile(source1, destination1)
            
            my_logger.info('cropped images and data.pkl moved into GMIC')

            project_id = 'project_id'
            topic_name =  'topic_name'
            publisher = pubsub.PublisherClient()
            topic_path = publisher.topic_path(project_id, topic_name)
            msg_info = json.dumps(msg_info)
            data = u"{}".format(msg_info)
            data = data.encode("utf-8")
            future = publisher.publish(topic_path, data )
            my_logger.info('information publish for GMIC and run_model (message acknowledged)')

            logs_bucket = 'logs_bucket'
            credential_json_file = "credential_json_file"
            storage_client = store.Client.from_service_account_json(credential_json_file)
            log_bucket = storage_client.get_bucket(logs_bucket)
            log_blob = log_bucket.blob('{0}/logs/{1}/{2}/stage-1-&-2'.format(userId,date,currentTime))
            log_blob.upload_from_filename('{}.log'.format(msg_id))
            #print(msg_id)
            os.remove('./{}.log'.format(msg_id))
            print("waiting for new message...")
            message.ack()

        except Exception as e:
            # if imges dim is very small to process model sending error msg accordingly
            my_logger.info('ERROR: {}'.format(e))
    
            invalid = {"Image_Predictions":{
                'error':'true',
                 'msg': "view mammography images size should be minimum 2677x1942 pixels and 2974x1748 pixels for CC and MLO"
                 }}
            visual_invalid ={"visualization":{
				"msg":"invalid image resolutions for visualizations",
                "error":"true"}}
			if (not len(firebase_admin._apps)):
            	cred = credentials.Certificate(credential_json_file)
                fa=firebase_admin.initialize_app(cred, {"databaseURL": databaseURL,'storageBucket':storageBucket})
                fc=firebase_admin.firestore.client(fa)
                db = firestore.client()
                doc_ref = db.collection(u'stripe_customers/{0}/results'.format(userId)).document(currentTime)
                doc_ref.update(invalid)
                doc_ref.update(visual_invalid)
             else:
                print('alredy initialize')
                db = firestore.client()
                doc_ref = db.collection(u'stripe_customers/{0}/results'.format(userId)).document(currentTime)
                doc_ref.update(invalid)
                doc_ref.update(visual_invalid)

       		my_logger.info("invalid images CC-MLO info saved in firebase")
            logs_bucket = logs_bucket
            credential_json_file = credential_json_file
            storage_client = store.Client.from_service_account_json(credential_json_file)
            log_bucket = storage_client.get_bucket(logs_bucket)
            log_blob = log_bucket.blob('{0}/logs/{1}/{2}/stage-1-&-2'.format(userId,date,currentTime))
            log_blob.upload_from_filename('{}.log'.format(msg_id))
            os.remove('./{}.log'.format(msg_id))
            shutil.rmtree("sample_output/{}".format(msg_id), ignore_errors=True)
            shutil.rmtree("sample_data/{}".format(msg_id), ignore_errors=True)
            message.ack()
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
