import pickle
import tensorflow as tf
import cv2
import numpy as np
import time
import logging
import argparse

from network_cmu import CmuNetwork
from common import estimate_pose, CocoPairsRender
from network_kakao import KakaoNetwork

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

config = tf.ConfigProto()
config.gpu_options.allocator_type = 'BFC'
config.gpu_options.per_process_gpu_memory_fraction = 0.95
config.gpu_options.allow_growth = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tensorflow Openpose Inference')
    parser.add_argument('--imgpath', type=str, default='./images/person3.jpg')
    parser.add_argument('--input-width', type=int, default=320)
    parser.add_argument('--input-height', type=int, default=240)
    parser.add_argument('--stage-level', type=int, default=6)
    parser.add_argument('--model', type=str, default='cmu', help='cmu(original) / kakao(faster version)')
    args = parser.parse_args()

    input_node = tf.placeholder(tf.float32, shape=(None, args.input_height, args.input_width, 3), name='image')

    with tf.Session(config=config) as sess:
        if args.model == 'kakao':
            net = KakaoNetwork({'image': input_node}, trainable=False, conv_width=1.0)
            net.load('./models/numpy/fastopenpose_coco_v170729.npy', sess)
        elif args.model == 'cmu':
            net = CmuNetwork({'image': input_node}, trainable=False)
            net.load('./models/numpy/openpose_coco.npy', sess)
        else:
            raise Exception('Invalid Mode.')

        logging.debug('read image+')
        image = cv2.imread(args.imgpath)
        image = cv2.resize(image, (args.input_width, args.input_height))
        image = image.astype(float)
        image -= 128.0
        image /= 128.0

        vec = sess.run(net.get_output(name='concat_stage7'), feed_dict={'image:0': [image]})

        logging.debug('inference+')
        a = time.time()
        pafMat, heatMat = sess.run(
            [
                net.get_output(name='Mconv7_stage{}_L1'.format(args.stage_level)),
                net.get_output(name='Mconv7_stage{}_L2'.format(args.stage_level))
            ], feed_dict={'image:0': [image]}
        )
        logging.info('inference- elapsed_time={}'.format(time.time() - a))
        a = time.time()
        pafMat, heatMat = sess.run(
            [
                net.get_output(name='Mconv7_stage{}_L1'.format(args.stage_level)),
                net.get_output(name='Mconv7_stage{}_L2'.format(args.stage_level))
            ], feed_dict={'image:0': [image]}
        )
        logging.info('inference- elapsed_time={}'.format(time.time() - a))
        a = time.time()
        pafMat, heatMat = sess.run(
            [
                net.get_output(name='Mconv7_stage{}_L1'.format(args.stage_level)),
                net.get_output(name='Mconv7_stage{}_L2'.format(args.stage_level))
            ], feed_dict={'image:0': [image]}
        )
        logging.info('inference- elapsed_time={}'.format(time.time() - a))

        heatMat, pafMat = heatMat[0], pafMat[0]
        heatMat = np.rollaxis(heatMat, 2, 0)
        pafMat = np.rollaxis(pafMat, 2, 0)

        logging.info('pickle data')
        with open('heatmat.pickle', 'wb') as pickle_file:
            pickle.dump(heatMat, pickle_file, pickle.HIGHEST_PROTOCOL)
        with open('pafmat.pickle', 'wb') as pickle_file:
            pickle.dump(pafMat, pickle_file, pickle.HIGHEST_PROTOCOL)

        logging.info('pose+')
        a = time.time()
        humans = estimate_pose(heatMat, pafMat)
        logging.info('pose- elapsed_time={}'.format(time.time() - a))

        # display
        image = cv2.imread(args.imgpath)
        image_h, image_w = image.shape[:2]
        heat_h, heat_w = heatMat[0].shape[:2]
        for _, human in humans.items():
            for part in human:
                if part['partIdx'] not in CocoPairsRender:
                    continue
                center1 = (int((part['c1'][0] + 0.5) * image_w / heat_w), int((part['c1'][1] + 0.5) * image_h / heat_h))
                center2 = (int((part['c2'][0] + 0.5) * image_w / heat_w), int((part['c2'][1] + 0.5) * image_h / heat_h))
                cv2.circle(image, center1, 2, (255, 0, 0), thickness=3, lineType=8, shift=0)
                cv2.circle(image, center2, 2, (255, 0, 0), thickness=3, lineType=8, shift=0)
                image = cv2.line(image, center1, center2, (255, 0, 0), 1)
        cv2.imshow('result', image)
        cv2.waitKey(0)
