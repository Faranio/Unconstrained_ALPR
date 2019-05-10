import sys, os
import cv2
import keras
import numpy as np
import traceback
import time
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

def russia(result1):
    result1.replace(" ", "")
    output = result1
    valid = True
    j = 0
    
    if len(result1) > 9 or len(result1) < 8:
        
        return ""
        
    for i in result1:
        if j == 0 or j == 4 or j == 5:
            if not ((ord(i) >= 65 and ord(i) <= 90) or (ord(i) >= 97 and ord(i) <= 122)):
                return ""
        if j == 1 or j == 2 or j == 3 or j == 6 or j == 7 or j == 8:
            if not (ord(i) >= 48 and ord(i) <= 57):
                return ""
        if j == 8:
            if not ((ord(i) >= 48 and ord(i) <= 57) or (ord(i) <= 32 and ord(i) >= 0)):
                return ""
        j += 1

    return output

def kazakhstan(result1):

    output = result1
    valid = True
    j = 0

    if len(result1) != 7:

        if len(result1) != 8:
            
            return ""
            
        for i in output:
            if j == 0 or j == 1 or j == 2 or j == 6:
                if not (ord(i) >= 48 and ord(i) <= 57):
                    return ""
            if j == 3 or j == 4 or j == 5:
                if not((ord(i) >= 65 and ord(i) <= 90) or (ord(i) >= 97 and ord(i) <= 122)):
                    return "" 
            if j == 7:
                if not (ord(i) >= 48 and ord(i) <= 57):
                    return ""
            j += 1

        return output

    else:
        
        for i in result1:
            if j == 0 or j == 4 or j == 5 or j == 6:
                if not ((ord(i) >= 65 and ord(i) <= 90) or (ord(i) >= 97 and ord(i) <= 122)):
                    return ""
            if j == 1 or j == 2 or j == 3:
                if not (ord(i) >= 48 and ord(i) <= 57):
                    return ""
            j += 1

        return output

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
		frame = frame[400:]

        if ret == False:
            break

        try:

            print("Frame number #" + str(n))

            if n % 5 == 0:
        
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

                        cv2.imwrite('%s/%s_%dcar.png' % (output_dir,n,i),Icar)

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

                    bname = splitext(basename(img_path))[0]
                    Ivehicle = cv2.imread(img_path)

                    ratio = float(max(Ivehicle.shape[:2]))/min(Ivehicle.shape[:2])
                    side  = int(ratio*288.)
                    bound_dim = min(side + (side%(2**4)),608)
                    print "\t\tBound dim: %d, ratio: %f" % (bound_dim,ratio)

                    Llp,LlpImgs,_ = detect_lp(wpod_net,im2single(Ivehicle),bound_dim,2**4,(240,80),lp_threshold)

                    if len(LlpImgs):
                        Ilp = LlpImgs[0]
                        Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
                        Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)

                        s = Shape(Llp[0].pts)

                        cv2.imwrite('%s/%s_lp.png' % (output_dir,n),Ilp*255.)
                        writeShapes('%s/%s_lp.txt' % (output_dir,n),[s])

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

                        result = russia(lp_str)

                        if (result == ""):
                            result = kazakhstan(lp_str)

                        if (result != ""):
                            print '-----------------------------------------------------------------'
                            print '\t\tLP: %s' % result
                            print '-----------------------------------------------------------------'

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

                        lp_label 		= '%s/%s_%dcar_lp.txt'		% (output_dir,n,i)
                        lp_label_str 	= '%s/%s_%dcar_lp_str.txt'	% (output_dir,n,i)

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

                shutil.rmtree('tmp/output/')

                cv2.imshow('Video', frame)

                if cv2.waitKey(1)&0xff==ord('q'):
                    break

                n += 1

                continue
            
            else:
                continue

        except:
            traceback.print_exc()
            sys.exit(1)

        sys.exit(0)

    print("Finished.")
    cap.release()
