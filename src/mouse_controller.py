'''
This is a sample class that you can use to control the mouse pointer.
It uses the pyautogui library. You can set the precision for mouse movement
(how much the mouse moves) and the speed (how fast it moves) by changing 
precision_dict and speed_dict.
Calling the move function with the x and y output of the gaze estimation model
will move the pointer.
'''
import pyautogui
pyautogui.FAILSAFE = False

class MouseController:
    def __init__(self, precision, speed):
        #precision_dict={'high':100, 'low':1000, 'medium':500}
        #speed_dict={'fast':1, 'slow':10, 'medium':5}
        precision_dict={'high':100, 'low':1000, 'medium':500}
        speed_dict={'fast':1, 'slow':5, 'medium':3}

        self.precision=precision_dict[precision]
        self.speed=speed_dict[speed]

    def move(self, x, y):   
        #Move depending of model coordinates
        pyautogui.moveRel(x*self.precision, -1*y*self.precision, duration=self.speed)
