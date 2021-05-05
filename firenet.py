################################################################################

# Example : perform live fire detection in video using FireNet CNN

# Copyright (c) 2017/18 - Andrew Dunnings / Toby Breckon, Durham University, UK

# License : https://github.com/tobybreckon/fire-detection-cnn/blob/master/LICENSE

################################################################################



import cv2
import os
import sys
import math
import subprocess
import ffmpeg
import redis

################################################################################
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tflearn
from tflearn.layers.core import *
from tflearn.layers.conv import *
from tflearn.layers.normalization import *
from tflearn.layers.estimator import regression

################################################################################

device_id=1

################################################################################

def construct_firenet (x,y, training=False):

    # Build network as per architecture in [Dunnings/Breckon, 2018]

    network = tflearn.input_data(shape=[None, y, x, 3], dtype=tf.float32)

    network = conv_2d(network, 64, 5, strides=4, activation='relu')

    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)

    network = conv_2d(network, 128, 4, activation='relu')

    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)

    network = conv_2d(network, 256, 1, activation='relu')

    network = max_pool_2d(network, 3, strides=2)
    network = local_response_normalization(network)

    network = fully_connected(network, 4096, activation='tanh')
    if(training):
        network = dropout(network, 0.5)

    network = fully_connected(network, 4096, activation='tanh')
    if(training):
        network = dropout(network, 0.5)

    network = fully_connected(network, 2, activation='softmax')

    # if training then add training hyperparameters

    if(training):
        network = regression(network, optimizer='momentum',
                            loss='categorical_crossentropy',
                            learning_rate=0.001)

    # constuct final model

    model = tflearn.DNN(network, checkpoint_path='firenet',
                        max_checkpoints=1, tensorboard_verbose=2)

    return model

################################################################################

if __name__ == '__main__':

################################################################################
    # TODO:
    communication = redis.Redis(host='39.101.200.0', port=6379, db=0, password='Wrj145325' )
    communication.set(device_id,'online')
    # device_num = communication.get('device_num')
    # communication.set('device_num',device_num+1)
    # communication.set(device_num+1,'online')

    # construct and display model

    model = construct_firenet(224, 224, training=False)
    print("Constructed FireNet ...")

    model.load(os.path.join("models/FireNet", "firenet"),weights_only=True)
    print("Loaded CNN network weights ...")

################################################################################

    # network input sizes

    rows = 224
    cols = 224

    # display and loop settings

    windowName = "Live Fire Detection - FireNet CNN"
    keepProcessing = True

################################################################################

    # rtmp server
    RTMP_SERVER = 'rtmp://127.0.0.1:1935/live/stream'

###############################################################################

    if len(sys.argv) == 2:

        # load video file from first command line argument

        video = cv2.VideoCapture(sys.argv[1])
        print("Loaded video ...")

        # create window

        cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)

        # get video properties

        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_time = round(1000/fps)
        sizeStr = str(width)+'x'+str(height)

        # way 1:
        '''
        command = ['ffmpeg.exe',
                   '-y',
                    '-f', 'rawvideo',
                    '-vcodec', 'rawvideo',
                    '-s', sizeStr,
                    '-pix_fmt', 'bgr24',
                    '-r', str(fps),
                    '-i', '-',
                    '-an',
                    '-c:v', 'libx264',
                    '-f', 'flv',
                    RTMP_SERVER
                  ]
               
        pipe = subprocess.Popen(command, stdin=subprocess.PIPE, shell=False)
        '''
        # way 2:
        process = (
            ffmpeg
            .input('pipe:',format='rawvideo',pix_fmt='bgr24',s='{}'.format(sizeStr),r='{}'.format(str(fps)),vcodec='rawvideo')
            .output(RTMP_SERVER,format='flv',vcodec='libx264')
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )


        while (keepProcessing):

            # start a timer (to see how long processing and display takes)

            start_t = cv2.getTickCount()

            # get video frame from file, handle end of file

            ret, frame = video.read()
            if not ret:
                print("... end of video file reached")
                break

            # re-size image to network input size and perform prediction

            small_frame = cv2.resize(frame, (rows, cols), cv2.INTER_AREA)

            # perform prediction on the image frame which is:
            # - an image (tensor) of dimension 224 x 224 x 3
            # - a 3 channel colour image with channel ordering BGR (not RGB)
            # - un-normalised (i.e. pixel range going into network is 0->255)

            output = model.predict([small_frame])

            # label image based on prediction

            if round(output[0][0]) == 1:
                communication.set(device_id, 'onfire')
                cv2.rectangle(frame, (0,0), (width,height), (0,0,255), 50)
                cv2.putText(frame,'FIRE',(int(width/16),int(height/4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),10,cv2.LINE_AA)
            else:
                communication.set(device_id, 'online')
                cv2.rectangle(frame, (0,0), (width,height), (0,255,0), 50)
                cv2.putText(frame,'CLEAR',(int(width/16),int(height/4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,255),10,cv2.LINE_AA)


            # stop the timer and convert to ms. (to see how long processing and display takes)

            stop_t = ((cv2.getTickCount() - start_t)/cv2.getTickFrequency()) * 1000

            # image display and key handling

            cv2.imshow(windowName, frame)
            # print(type(frame))
            # way 1 :
            # pipe.stdin.write(frame.tobytes())
            process.stdin.write(frame.astype(np.uint8).tobytes())




            # wait fps time or less depending on processing time taken (e.g. 1000ms / 25 fps = 40 ms)

            key = cv2.waitKey(max(2, frame_time - int(math.ceil(stop_t)))) & 0xFF
            if key == ord('x'):
                keepProcessing = False
            elif key == ord('f'):
                cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                pass
        # pipe.terminate()

    else:
        print("usage: python firenet.py videofile.ext")

    communication.set(device_id, 'offline')

################################################################################

