import numpy as np
import cv2 
from openvino.inference_engine import IENetwork, IECore

class FaceDetection:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        self.model_weights=model_name+'.bin'
        self.model_structure=model_name+'.xml'
        self.device= device
        self.threshold= 0.5
        self.request_id=0
        self.num_requests=1 
        self.ie = IECore()
        self.net = None
        self.exec_net = None
        self.frame_width = None
        self.frame_height = None
        self.extensions = extensions

        self.check_model()
        self.input_name=next(iter(self.net.inputs))
        self.input_shape=self.net.inputs[self.input_name].shape
        self.output_name=next(iter(self.net.outputs))
        self.output_shape=self.net.outputs[self.output_name].shape

    def check_model(self):
        try:
            self.net=self.ie.read_network(self.model_structure, self.model_weights)
        except:
            raise ValueError("Could not Initialise the network. Have you enterred the correct model path?")

    def load_model(self):
        self.exec_net = self.ie.load_network(network=self.net, device_name=self.device, num_requests=self.num_requests)

    def predict(self, image):
        self.frame_width, self.frame_height = image.shape[1], image.shape[0]
        
        preprocessed_image = self.preprocess_input(image)
        input_dict={self.input_name: preprocessed_image}

        self.exec_net.requests[self.request_id].infer(input_dict)
        outputs = self.exec_net.requests[self.request_id].outputs[self.output_name]

        coords = self.preprocess_output(outputs)
        
        self.draw_outputs(coords, image)
        return coords, image

    def draw_outputs(self, coords, image):
        color = (255, 0, 0) 
        thickness = 4
        for box in coords:
            cv2.rectangle(image, (box[0],box[1]), (box[2], box[3]), color, thickness)
            cv2.putText(image, 'Face', (box[0]+2,box[1]+12), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2)

    def preprocess_input(self, image):
        input_img = cv2.resize(image, (self.input_shape[3], self.input_shape[2]), 
                               interpolation = cv2.INTER_AREA) 
        input_img = np.moveaxis(input_img, -1, 0)
        return input_img
 

    def preprocess_output(self, outputs):
        coords = []
    
        for obj in outputs[0][0]:
            if obj[2] > self.threshold:
                xmin = int(obj[3]* self.frame_width)
                ymin = int(obj[4]* self.frame_height)
                xmax = int(obj[5]* self.frame_width)
                ymax = int(obj[6]* self.frame_height)
                coords.append([xmin, ymin, xmax, ymax])
        return coords
