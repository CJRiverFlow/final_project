import cv2
import numpy as np
from openvino.inference_engine import IENetwork, IECore

class HeadPoseModel:
    '''
    Class for the Head Pose Estimation Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device= device
        self.request_id=0
        self.num_requests=1 
        self.ie = IECore()
        self.net = None
        self.exec_net = None
        self.extensions = extensions

        self.check_model()
        self.input_name=next(iter(self.net.inputs))
        self.input_shape=self.net.inputs[self.input_name].shape
        

    def check_model(self):
        try:
            self.net=self.ie.read_network(self.model_structure, self.model_weights)
        except:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")


    def load_model(self):
        self.exec_net = self.ie.load_network(network=self.net, 
                        device_name=self.device, num_requests=self.num_requests)


    def predict(self, image):
        
        preprocessed_image = self.preprocess_input(image)
        input_dict={self.input_name: preprocessed_image}

        self.exec_net.requests[self.request_id].infer(input_dict)

        yaw = self.exec_net.requests[self.request_id].outputs["angle_y_fc"]
        pitch = self.exec_net.requests[self.request_id].outputs["angle_p_fc"]
        roll = self.exec_net.requests[self.request_id].outputs["angle_r_fc"]
        
        angles = np.array([yaw[0][0], pitch[0][0], roll[0][0]]).reshape(1,3)

        return angles


    def preprocess_input(self, image):
        input_img = cv2.resize(image, (self.input_shape[3], self.input_shape[2]), 
                                interpolation = cv2.INTER_AREA) 
        input_img = np.moveaxis(input_img, -1, 0)
        return input_img

