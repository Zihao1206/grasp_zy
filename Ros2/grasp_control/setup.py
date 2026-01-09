from setuptools import setup

package_name = 'grasp_control'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jet',
    maintainer_email='jet@zoneyung.com',
    description='机械臂和夹爪控制节点',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'arm_controller = grasp_control.arm_controller:main',
            'gripper_server = grasp_control.gripper_server:main',
        ],
    },
)

