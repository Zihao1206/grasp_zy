# from networkx import authority_matrix
from tkinter import messagebox
from robotic_arm import *
import socket
import tkinter as tk
from tkinter import ttk
import ctypes

ip = "192.168.1.18"
port = 8080
robot = Arm(RM65, ip, port)
serve_address = (ip,port)
 
def Socket_Connect():
    client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        client.connect(serve_address)
        return True
    except socket.error as e:
        return False
    
def Motor_Setup():
    Socket_Connect()
    robot.Set_Tool_Voltage(3) # Set Voltage
    robot.Set_Modbus_Mode(1, 9600, 1) # Set Modbus Mode
    # robot.Write_Single_Register(port=1, device=0x01, address=0x2a, data=0x1005) # Set Zero Position
    

def Rotate_Backward():
    robot.Write_Single_Register(port=1, device=0x01, address=0x2a, data=0x0000) # Motor rotate backward

def Rotate_Forward():
    robot.Write_Single_Register(port=1, device=0x01, address=0x2a, data=0x0010) # Motor rotate forward

def Rotate_Backward_Autostop():
    robot.Write_Single_Register(port=1, device=0x01, address=0x2a, data=0x0001) # Motor rotate backward until the max level

def Rotate_Forward_Autostop():
    robot.Write_Single_Register(port=1, device=0x01, address=0x2a, data=0x0012) # Motor rotate forward until the min level

def Move_To(position: int):
    pos_1 = 0b00000000
    if position < 0:
        position = -position
        pos_1 = 0b10000000
    pos_2 = (position >> 16) & 0b11111111
    pos_3 = (position >> 8) & 0b11111111
    pos_4 = position & 0b11111111
    robot.Write_Registers(port=1, device=0x01, address=0x2b, num=0x0002, single_data=[pos_1, pos_2, pos_3, pos_4])
    
def Stop(hardstop=True):
    if hardstop:
        robot.Write_Single_Register(port=1, device=0x01, address=0x2d, data=0x0000) # Stop motor (cannot be moved manually)
    else:
        robot.Write_Single_Register(port=1, device=0x01, address=0x2d, data=0x0001)

def Motor_Close():
    robot.Set_Tool_Voltage(0)
    robot.Close_Modbus_Mode(1, False)
    
def Set_Velocity(v: int):
    v_1 = (v >> 24) & 0b11111111
    v_2 = (v >> 16) & 0b11111111
    v_3 = (v >> 8) & 0b11111111
    v_4 = v & 0b11111111
    robot.Write_Registers(port=1, device=0x01, address=0x26, num=0x0002, single_data=[v_1, v_2, v_3, v_4])
 
def on_button_s_click():
    if hardflag.get() == 1:
        Stop(True)
    else:
        Stop(False)
        

# -------- Below is the GUI part --------

ctypes.windll.shcore.SetProcessDpiAwareness(1)

root = tk.Tk()

hardflag = tk.IntVar()

root.title("Motor Debugger")

button_1 = ttk.Button(root, text="Motor Setup (24V)")
button_1.bind("<ButtonPress-1>", lambda event: Motor_Setup())
button_1.grid(row=0, column=0, padx=10, pady=10)

button_r = ttk.Button(root, text="Release (Press A)")
button_r.bind("<ButtonPress-1>", lambda event: Rotate_Backward())
button_r.bind("<ButtonRelease-1>", lambda event: Stop())
button_r.grid(row=0, column=1, padx=10, pady=10)

button_g = ttk.Button(root, text="Grasp (Press D)")
button_g.bind("<ButtonPress-1>", lambda event: Rotate_Forward())
button_g.bind("<ButtonRelease-1>", lambda event: Stop())
button_g.grid(row=0, column=2, padx=10, pady=10)

button_s = ttk.Button(root, text="Stop")
button_s.bind("<ButtonPress-1>", lambda event: on_button_s_click)
button_s.grid(row=1, column=0, padx=10, pady=10)

button_2 = ttk.Button(root, text="Motor Close (0V)")
button_2.bind("<ButtonPress-1>", lambda event: Motor_Close())
button_2.grid(row=1, column=1, padx=10, pady=10)

textbox = ttk.Entry(root, width=10)
textbox.grid(row=2, column=0, padx=10, pady=10)

user_input_pos = 0

def on_submit_position():
    try:
        user_input_pos = int(textbox.get())
        if user_input_pos < -16777215 or user_input_pos > 16777215:
            messagebox.showerror("Error", "The input value must be in [-16777215, 16777215]. The change is not applied.")
        else:
            Move_To(user_input_pos)
    except ValueError:
        messagebox.showerror("Error", "The input value must be an integer. The change is not applied.")
    
button_submit_p = ttk.Button(root, text="Move To", command=on_submit_position)
button_submit_p.grid(row=2, column=1, padx=10, pady=10)
    
textbox2 = ttk.Entry(root, width=10)
textbox2.grid(row=3, column=0, padx=10, pady=10)

def on_submit_velocity():
    try:
        user_input_v = float(textbox2.get())
        if user_input_v <= 0 or user_input_v > 10:
            messagebox.showerror("Error", "The input value must be in (0, 10]. The change is not applied.")
        else:
            Set_Velocity(int(user_input_v * 51200))
    except ValueError:
        messagebox.showerror("Error", "The input value must be a number. The change is not applied.")
    
button_submit_v = ttk.Button(root, text="Set Velocity (rolls per second)", command=on_submit_velocity)
button_submit_v.grid(row=3, column=1, padx=10, pady=10)
    
def on_closing():
    Motor_Close()
    root.destroy()

def on_key_press(event):
    if event.keysym == 'a':
        Rotate_Backward()
    elif event.keysym == 'd':
        Rotate_Forward()
        
def on_key_release(event):
    if event.keysym == 'a' or event.keysym == 'd':
        Stop()
        
root.bind('<KeyPress>', on_key_press)
root.bind('<KeyRelease>', on_key_release)

root.protocol("WM_DELETE_WINDOW", lambda: on_closing())

root.mainloop()