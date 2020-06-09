import cv2
import time
import pandas as pd
import pyautogui

from src.input_feeder import InputFeeder
from src.face_detection import FaceDetection
from src.facial_landmarks_detection import FacialLandmarks
from src.head_pose_estimation import HeadPoseModel
from src.gaze_estimation import GazeEstimationModel
from src.mouse_controller import MouseController
from argparse import ArgumentParser

video_path = "bin/demo.mp4"
models_path = "models/intel/"

#Dataframe for saving perfomance times
df = pd.DataFrame(0.0, columns=["face_detection","face_landmarks","headpose_estimation",
                  "gaze_estimation"],index=["loading_time","inference_time"])


def build_argparser():
    """
    Parse command line arguments.
    Return parser
    """
    parser = ArgumentParser()

    parser.add_argument("--input_type", required=True, type=str,
                        default="cam",
                        help="Type of video input, camera or file")
    parser.add_argument("--input_file", required=False, type=str,
                        default="bin/demo.mp4",
                        help="Path to video file to be used")
    parser.add_argument("--speed", required=False, type=str,
                        default="medium",
                        help="How much the mouse moves")
    parser.add_argument("--precision", required=False, type=str,
                        default="medium",
                        help="How fast the mouse pointer moves")
    parser.add_argument("--model_precision", required=False, type=str,
                        default="FP32",
                        help="Model precision options: FP32, FP16, INT8")
    parser.add_argument("--display_values", required=False, type=str,
                        default="False",
                        help="Flag to display intermediate model outputs")
    return parser


def declare_models(selected_precision):
    """
    Select model based on model precision input
    """
    global fd, fl, hp, gaze

    valid_values = {"FP32":"FP32",
                    "FP16":"FP16",
                    "INT8":"FP16-INT8"}
    
    #Unique option for face detection.
    fd_model = "face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001"

    #Applying user selection.
    if selected_precision in valid_values:
        fland_model = "landmarks-regression-retail-0009/{}/landmarks-regression-retail-0009"\
                    .format(valid_values[selected_precision])
        head_pose = "head-pose-estimation-adas-0001/{}/head-pose-estimation-adas-0001"\
                    .format(valid_values[selected_precision])
        gaze_model = "gaze-estimation-adas-0002/{}/gaze-estimation-adas-0002"\
                    .format(valid_values[selected_precision])
    else:
        raise ValueError('No valid model precision seleted')
    
    fd = FaceDetection(models_path+fd_model)
    fl = FacialLandmarks(models_path+fland_model) 
    hp = HeadPoseModel(models_path+head_pose)
    gaze = GazeEstimationModel(models_path+gaze_model)

def print_values(output_image,value_dic):
    y_pos = 0
    for value in value_dic:
        text = value+str(value_dic[value])
        y_pos+=20
        cv2.putText(output_image, str(text), (20,y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1) 

def get_mouse_coordinates(batch, display_values):
    """
    Run model inference pipeline
    Return x, y coordinates from gaze estimation model
    """
    output = batch.copy()
    global df

    #GETTING THE FACE and measure time. 
    fd_start_inference_time=time.time()
    face_coords, image = fd.predict(batch) #inference
    df["face_detection"]["inference_time"] += round(time.time()-fd_start_inference_time,3)

    ##CROPPING THE FACE - set to first one detected. 
    cropped_face_img = image[face_coords[0][1]:face_coords[0][3], 
                        face_coords[0][0]:face_coords[0][2]]
    
    #GETTING CROPPED EYES and measure time.
    fl_start_inference_time=time.time()
    landmarks, left_eye, right_eye = fl.predict(cropped_face_img) #inference
    df["face_landmarks"]["inference_time"] += round(time.time()-fl_start_inference_time,3)
    
    #GETTING POSE ANGLES and measure time.
    hp_start_inference_time=time.time()
    head_pose_angles = hp.predict(cropped_face_img) #inference
    df["headpose_estimation"]["inference_time"] += round(time.time()-hp_start_inference_time,3)
    #print(head_pose_output)

    #RUNNING GAZE ESTIMATION and measure time.
    gaze_start_inference_time=time.time()
    gaze_outputs = gaze.predict(left_eye, right_eye, head_pose_angles) #inference
    df["gaze_estimation"]["inference_time"] += round(time.time()-gaze_start_inference_time,3)
    
    #Printing values on screen
    #print("gaze model outputs: ",gaze_outputs)
    output = cv2.resize(output, (1080, 600), interpolation = cv2.INTER_AREA)
    if display_values == "True":
        value_dic = {"Face coordinates: ": face_coords, 
                    "Face landmarks (eyes only): ":landmarks,
                    "Head pose angels: ":head_pose_angles,
                    "Gaze estimation values: ":gaze_outputs,
                    "Accum inference time(s): ": round(df.iloc[1,:].values.sum(),3)}
        print_values(output,value_dic)
    cv2.imshow("Frame",output)
    cv2.waitKey(1)

    return gaze_outputs[0][0], gaze_outputs[0][1]


def main():
    args = build_argparser().parse_args()
    declare_models(args.model_precision)
    global df 
    
    #LOADING MODELS and BENCHMARKING
    #Face detection  
    fd_start_load_time=time.time()
    fd.load_model()
    df["face_detection"]["loading_time"] = round(time.time()-fd_start_load_time,3)

    #Facial landmarks detection
    fl_start_load_time=time.time()
    fl.load_model()
    df["face_landmarks"]["loading_time"] = round(time.time()-fl_start_load_time,3)

    #Head pose estimation
    hp_start_load_time=time.time() 
    hp.load_model()
    df["headpose_estimation"]["loading_time"] = round(time.time()-hp_start_load_time,3)

    #Gaze estimation model
    gaze_start_load_time=time.time() 
    gaze.load_model()
    df["gaze_estimation"]["loading_time"] = round(time.time()-gaze_start_load_time,3)
    

    #CUSTOM FEATURE
    #Move mouse to specific screen point to watch better behaviour
    print('Setting cursor to screen start position...')
    screen_res = pyautogui.size()
    pyautogui.moveTo(int(screen_res[0]/4), int(screen_res[1]/4), duration=1)  

    #Setting mouse controller
    controller = MouseController(precision=args.precision, speed=args.speed)

    feed=InputFeeder(input_type=args.input_type, input_file=args.input_file)
    feed.load_data()
    for batch in feed.next_batch():
        if batch is not None: 
            #MOVING THE MOUSE
            x_coord, y_coord  = get_mouse_coordinates(batch, args.display_values)
            controller.move(x_coord,y_coord)
        else:
            feed.close()
            break
    
    #Saving perfomance stats
    df = round(df,3)
    saving_path = "stats_{}.csv".format(args.model_precision)
    df.to_csv(saving_path)
    print("Stats saved as: ", saving_path)

    print("Done")

if __name__ == "__main__":
    main()