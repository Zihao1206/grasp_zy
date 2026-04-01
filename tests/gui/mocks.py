"""
Mock interfaces for GUI testing.
"""
import numpy as np

class MockCamera:
    """Mock camera that returns fake depth and color images."""
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
    
    def get_img(self):
        depth_image = np.random.randint(0, 65536, (self.height, self.width), dtype=np.uint16)
        color_image = np.random.randint(0, 256, (self.height, self.width, 3), dtype=np.uint8)
        return depth_image, color_image
    
    def stop(self):
        pass

class MockRobot:
    """Mock robot arm interface."""
    def rm_set_arm_stop(self):
        return 0

class MockGripper:
    """Mock gripper interface."""
    def gripper_position(self, position):
        pass
    def gripper_initial(self):
        pass

class MockGrasp:
    """Mock Grasp class for GUI testing."""
    def __init__(self, hardware=False):
        self.robot_speed = 30
        self.camera = MockCamera()
        self.robot = MockRobot()
        self.gripper = MockGripper()
    
    def obj_grasp(self, label, vis=False):
        return True
    
    def init_gripper(self):
        self.gripper.gripper_initial()
    
    def detect_obj(self, label):
        return 1

OBJECT_LABELS = ['terminal', 'limit', 'voltage', 'soap', 'banana', 'carrot', 'daikon', 'relay']
