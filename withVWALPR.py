import sys, os
import cv2
import keras
import numpy as np
import traceback
import time
import DetectPlates
import DetectChars
import shutil

import darknet.python.darknet as dn

from src.label 					import Label, lwrite, Shape, writeShapes, dknet_label_conversion, lread, readShapes
from os.path 					import splitext, basename, isdir, isfile
from os 						import makedirs
from src.utils 					import crop_region, image_files_from_folder, im2single, nms
from darknet.python.darknet 	import detect
from glob 						import glob
from src.keras_utils 			import load_model, detect_lp
from src.drawing_utils			import draw_label, draw_losangle, write2img

from pdb import set_trace as pause

vehicle_threshold = .5

vehicle_weights = 'data/vehicle-detector/yolo-voc.weights'
vehicle_netcfg  = 'data/vehicle-detector/yolo-voc.cfg'
vehicle_dataset = 'data/vehicle-detector/voc.data'

vehicle_net  = dn.load_net(vehicle_netcfg, vehicle_weights, 0)
vehicle_meta = dn.load_meta(vehicle_dataset)

lp_threshold = .5

wpod_net_path = "data/lp-detector/wpod-net_update1.h5"
wpod_net = load_model(wpod_net_path)

def adjust_pts(pts,lroi):
	return pts*lroi.wh().reshape((2,1)) + lroi.tl().reshape((2,1))

"""with open('./config.json') as f:
    config = json.load(f)

ip = str(config["ip"])"""

if __name__ == "__main__":
    
    cap = cv2.VideoCapture("VIDEO4.avi")
    n = -1
    previous = ""

    while True:

        n += 1

        ret, frame = cap.read()

        if ret == False:
            break

        try:

            print("Frame number #" + str(n))

            if n % 3 == 0:
        
                cv2.imwrite('frame.jpg', frame)

    #################################   VEHICLE DETECTION   #################################

                #Time 1
                start_time = time.time()

                input_dir = 'frame.jpg'
                output_dir = 'tmp/output'

                if not isdir(output_dir):
                    makedirs(output_dir)

                print 'Searching for vehicles using YOLO...'

                print 'Scanning frame #%d' % n

                R,_ = detect(vehicle_net, vehicle_meta, input_dir, thresh=vehicle_threshold)

                R = [r for r in R if r[0] in ['car','bus']]

                if len(R) == 0:
                    continue

                print '\t\t%d cars found' % len(R)

                if len(R):

                    Iorig = frame
                    WH = np.array(Iorig.shape[1::-1],dtype=float)
                    Lcars = []

                    for i,r in enumerate(R):

                        cx,cy,w,h = (np.array(r[2])/np.concatenate( (WH,WH) )).tolist()
                        tl = np.array([cx - w/2., cy - h/2.])
                        br = np.array([cx + w/2., cy + h/2.])
                        label = Label(0,tl,br)
                        Icar = crop_region(Iorig,label)

                        Lcars.append(label)

                        cv2.imwrite('%s/%s_car.png' % (output_dir,n),Icar)

                    lwrite('%s/%s_cars.txt' % (output_dir,n),Lcars)

                end_time = time.time()
                print("Vehicle detection time: " + str(end_time - start_time))

    #################################   LICENSE PLATE DETECTION   #################################

                #Time 2
                start_time = time.time()

                input_dir  = 'tmp/output'
                output_dir = input_dir

                imgs_paths = glob('%s/*car.png' % input_dir)

                print 'Searching for license plates using WPOD-NET'

                for i,img_path in enumerate(imgs_paths):

                    print '\t Processing %s' % img_path

                    imgOriginalScene = cv2.imread(img_path)

                    listOfPossiblePlates = DetectPlates.detectPlatesInScene(imgOriginalScene)           # detect plates
                    listOfPossiblePlates = DetectChars.detectCharsInPlates(listOfPossiblePlates)        # detect chars in plates

                    if len(listOfPossiblePlates) == 0:                          # if no plates were found
                        #print("\nNO license plates were detected!\n")  # inform user no plates were found
                        continue

                    else:
                        #cv2.imwrite("detected.jpg", frame)                                                       # else
                                # if we get in here list of possible plates has at leat one plate

                                # sort the list of possible plates in DESCENDING order (most number of chars to least number of chars)
                        listOfPossiblePlates.sort(key = lambda possiblePlate: len(possiblePlate.strChars), reverse = True)

                                # suppose the plate with the most recognized chars (the first plate in sorted by string length descending order) is the actual plate
                        licPlate = listOfPossiblePlates[0]

                        Ilp = licPlate.imgPlate
                        Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
                        Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)

                        #s = Shape(Llp[0].pts)

                        cv2.imwrite('%s/%s_lp.png' % (output_dir,n),licPlate.imgPlate)
                        #writeShapes('%s/%s_lp.txt' % (output_dir,n),[s])

                end_time = time.time()
                print("License plate detection time: " + str(end_time - start_time))

    #################################   LICENSE PLATE OCR   #################################

                #Time 3
                start_time = time.time()

                input_dir  = 'tmp/output'
                output_dir = input_dir

                ocr_threshold = .4

                ocr_weights = 'data/ocr/ocr-net.weights'
                ocr_netcfg  = 'data/ocr/ocr-net.cfg'
                ocr_dataset = 'data/ocr/ocr-net.data'

                ocr_net  = dn.load_net(ocr_netcfg, ocr_weights, 0)
                ocr_meta = dn.load_meta(ocr_dataset)

                imgs_paths = sorted(glob('%s/*lp.png' % output_dir))

                print 'Performing OCR...'

                for i,img_path in enumerate(imgs_paths):

                    print '\tScanning %s' % img_path

                    bname = basename(splitext(img_path)[0])

                    R,(width,height) = detect(ocr_net, ocr_meta, img_path ,thresh=ocr_threshold, nms=None)

                    if len(R):

                        L = dknet_label_conversion(R,width,height)
                        L = nms(L,.45)

                        L.sort(key=lambda x: x.tl()[0])
                        lp_str = ''.join([chr(l.cl()) for l in L])

                        with open('%s/%s_str.txt' % (output_dir,n),'w') as f:
                           f.write(lp_str + '\n')

                        print '\t\tLP: %s' % lp_str

                    else:

                        print 'No characters found'	

                end_time = time.time()
                print("License plate OCR time: " + str(end_time - start_time))

    #################################   GENERATE OUTPUTS   #################################

                #Time 4
                start_time = time.time()

                YELLOW = (  0,255,255)
                RED    = (  0,  0,255)

                input_dir = 'frame.jpg'
                output_dir = 'tmp/output'

                I = cv2.imread(input_dir)

                detected_cars_labels = '%s/%s_cars.txt' % (output_dir, n)

                Lcar = lread(detected_cars_labels)

                sys.stdout.write('%s' % n)

                if Lcar:

                    for i,lcar in enumerate(Lcar):

                        draw_label(I,lcar,color=YELLOW,thickness=3)

                        lp_label 		= '%s/%s_car_lp.txt'		% (output_dir,n)
                        lp_label_str 	= '%s/%s_car_lp_str.txt'	% (output_dir,n)

                        if isfile(lp_label):

                            Llp_shapes = readShapes(lp_label)
                            pts = Llp_shapes[0].pts*lcar.wh().reshape(2,1) + lcar.tl().reshape(2,1)
                            ptspx = pts*np.array(I.shape[1::-1],dtype=float).reshape(2,1)
                            draw_losangle(I,ptspx,RED,3)

                            if isfile(lp_label_str):
                                with open(lp_label_str,'r') as f:
                                    lp_str = f.read().strip()
                                llp = Label(0,tl=pts.min(1),br=pts.max(1))
                                write2img(I,llp,lp_str)

                                sys.stdout.write(',%s' % lp_str)

                cv2.imwrite('%s/%s_output.png' % (output_dir,n),I)
                sys.stdout.write('\n')

                end_time = time.time()
                print("Generate outputs time: " + str(end_time - start_time))

            else:

                continue

            cv2.imshow('Video', frame)
            #Main.main(frame2)

            if cv2.waitKey(1)&0xff==ord('q'):
                break

            try:

                shutil.rmtree('tmp/output')

            except: pass

            continue

        except:
            traceback.print_exc()
            sys.exit(1)

        sys.exit(0)

    print("Finished.")
    cap.release()
