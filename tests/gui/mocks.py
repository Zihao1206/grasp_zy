import numpy as np


class MockCamera:
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
    def rm_set_arm_stop(self):
        return 0


class MockGripper:
    def gripper_position(self, position):
        pass

    def gripper_initial(self):
        pass


class MockGrasp:
    def __init__(self, hardware=False):
        self.robot_speed = 30
        self.camera = MockCamera()
        self.robot = MockRobot()
        self.gripper = MockGripper()
        self._grasp_plan = None

    def plan_grasp(self, label, vis_callback=None):
        self._grasp_plan = {'mock': True}
        if vis_callback is not None:
            vis_callback({
                'bboxes': [[100, 100, 200, 200]],
                'labels': [0],
                'classes': OBJECT_LABELS,
                'grasp_rect': [[100, 100], [200, 100], [200, 200], [100, 200]],
                'grasp_center': (150, 150),
                'crop_offset': 80,
                'target_label': label,
            })
        return True

    def execute_grasp(self, resume_event=None, cancel_event=None):
        if cancel_event and cancel_event.is_set():
            return False
        return True

    def obj_grasp(self, label, vis=False, vis_callback=None, **kwargs):
        if not self.plan_grasp(label, vis_callback=vis_callback):
            return False
        return self.execute_grasp()

    def init_gripper(self):
        self.gripper.gripper_initial()

    def detect_obj(self, label):
        return 1


OBJECT_LABELS = ['terminal', 'limit', 'voltage', 'soap', 'banana', 'carrot', 'daikon', 'relay']
