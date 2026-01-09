from tkinter import messagebox
from robotic_arm_package.robotic_arm import *
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


# -------- Below is the GUI part --------
    

DEFAULT_VELOCITY = 30
    
def Joint_6_Rotate(step):
    robot.Joint_Step_Cmd(num=6, step=step, v=DEFAULT_VELOCITY, block=False)

def Joint_6_Rotate_Still(dire=1):
    robot.Joint_Teach_Cmd(num=6, direction=dire, v=DEFAULT_VELOCITY, block=False)
    
Socket_Connect()

ctypes.windll.shcore.SetProcessDpiAwareness(1)

root = tk.Tk()

root.geometry("800x300")

velocity = tk.IntVar()

textbox1 = ttk.Entry(root, width=10)



def on_click_button1():
    try:
        input_step = float(textbox1.get())
        if abs(input_step) > 100:
            messagebox.showerror("Error", "Step value must be between -100 and 100")
        else:
            Joint_6_Rotate(input_step)
    except ValueError:
        messagebox.showerror("Error", "Please enter a valid number.")
        
def on_click_button2():
    Joint_6_Rotate_Still(0)

def on_release_button2():
    robot.Teach_Stop_Cmd()
    
def on_click_button3():
    Joint_6_Rotate_Still(1)
        
def on_release_button3():
    robot.Teach_Stop_Cmd()
    
def on_click_button4():
    robot.Move_Stop_Cmd()
    

ttk.Label(root, text="Step", font=("Microsoft Yahei", 9)).grid(row=0, column=0, padx=10, pady=10)

textbox1.grid(row=0, column=1, padx=10, pady=10)

button1 = ttk.Button(root, text="Rotate To", command=on_click_button1)
button1.grid(row=0, column=2, padx=10, pady=10)

ttk.Label(root, text="Positive: Anti-clockwise  Negative: Clockwise", font=("Microsoft Yahei", 9)).grid(row=1, column=1, padx=10, pady=10)

button2 = ttk.Button(root, text="Roll (Clockwise)")
button2.bind("<ButtonPress-1>", lambda event: on_click_button2())
button2.bind("<ButtonRelease-1>", lambda event: on_release_button2())
button2.grid(row=2, column=0, padx=10, pady=10)

button3 = ttk.Button(root, text="Roll (Anti-clockwise)")
button3.bind("<ButtonPress-1>", lambda event: on_click_button3())
button3.bind("<ButtonRelease-1>", lambda event: on_release_button3())
button3.grid(row=2, column=1, padx=10, pady=10)

button4 = ttk.Button(root, text="Stop")
button4.bind("<ButtonPress-1>", lambda event: on_click_button4())
button4.grid(row=2, column=2, padx=10, pady=10)

root.mainloop()