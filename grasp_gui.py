import tkinter as tk
from tkinter import font
import threading
import time
import subprocess
import os


# --- Configuration & Aesthetics ---
COLOR_BG = "#0a0e27"      # Deep Blue-Black
COLOR_BG_2 = "#1a1f3a"    # Lighter Blue
COLOR_FG = "#E0E0E0"      # Off-white (Text)
COLOR_ACCENT = "#00d9ff"  # Bright Cyan
COLOR_ACCENT_2 = "#7b2ff7" # Electric Purple
COLOR_ACCENT_3 = "#ff006e" # Hot Pink
COLOR_BUTTON_BG = "#1a1f3a" # Dark Blue
COLOR_BUTTON_HOVER = "#2a3f5a" # Lighter Blue
COLOR_BUTTON_ACTIVE = "#00d9ff" # Cyan
COLOR_WARNING = "#ff006e" # Hot Pink
COLOR_SUCCESS = "#00ff88" # Bright Green

FONT_FAMILY_HEADER = "Arial" # Fallback if specific fonts aren't available
FONT_FAMILY_MONO = "Courier New"

# Item List
ITEMS = ['terminal', 'limit', 'voltage', 'soap', 'banana', 'carrot', 'daikon', 'relay']

class HighTechButton(tk.Canvas):
    """Custom button with glow effects and animations"""
    def __init__(self, master, text, command, width=140, height=50):
        super().__init__(master, width=width, height=height, bg=COLOR_BG, highlightthickness=0)
        self.command = command
        self.text = text
        self.width = width
        self.height = height
        
        # Outer glow (initially invisible)
        self.glow = self.create_rectangle(0, 0, width, height, outline=COLOR_ACCENT, width=0, fill="")
        
        # Main button rectangle with gradient effect (simulated with overlays)
        self.rect = self.create_rectangle(3, 3, width-3, height-3, outline=COLOR_ACCENT, width=2, fill=COLOR_BUTTON_BG)
        
        # Inner highlight
        self.highlight = self.create_rectangle(6, 6, width-6, 10, outline="", fill=COLOR_ACCENT, stipple="gray25")
        
        # Text with shadow
        self.text_shadow = self.create_text(width/2+2, height/2+2, text=text, fill="#000000", font=(FONT_FAMILY_MONO, 11, "bold"))
        self.text_id = self.create_text(width/2, height/2, text=text, fill=COLOR_ACCENT, font=(FONT_FAMILY_MONO, 11, "bold"))
        
        # Corner decorations
        corner_size = 12
        self.create_line(3, corner_size, 3, 3, corner_size, 3, fill=COLOR_ACCENT_2, width=2)
        self.create_line(width-3, corner_size, width-3, 3, width-corner_size, 3, fill=COLOR_ACCENT_2, width=2)
        self.create_line(3, height-corner_size, 3, height-3, corner_size, height-3, fill=COLOR_ACCENT_2, width=2)
        self.create_line(width-3, height-corner_size, width-3, height-3, width-corner_size, height-3, fill=COLOR_ACCENT_2, width=2)

        # Bind events
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)
        self.bind("<Button-1>", self.on_click)

    def on_enter(self, event):
        self.itemconfig(self.glow, outline=COLOR_ACCENT_2, width=3)
        self.itemconfig(self.rect, fill=COLOR_BUTTON_HOVER, outline=COLOR_ACCENT_2, width=2)
        self.itemconfig(self.text_id, fill=COLOR_SUCCESS)

    def on_leave(self, event):
        self.itemconfig(self.glow, width=0)
        self.itemconfig(self.rect, fill=COLOR_BUTTON_BG, outline=COLOR_ACCENT, width=2)
        self.itemconfig(self.text_id, fill=COLOR_ACCENT)

    def on_click(self, event):
        self.itemconfig(self.rect, fill=COLOR_BUTTON_ACTIVE, outline=COLOR_SUCCESS)
        self.itemconfig(self.text_id, fill="#ffffff")
        self.update_idletasks()
        time.sleep(0.1)
        self.itemconfig(self.rect, fill=COLOR_BUTTON_HOVER)
        if self.command:
            self.command()

class RobotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("NEURAL LINK TERMINAL // ROBOT CONTROLLER")
        self.root.geometry("800x600")
        self.root.configure(bg=COLOR_BG)
        
        # Python executable path (use conda environment)
        self.python_exe = "/home/zh/anaconda3/envs/grasp_zy_py310/bin/python"
        # Fallback to system python if conda env not found
        if not os.path.exists(self.python_exe):
            self.python_exe = "python3"
        
        # Script path
        self.script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grasp_zy_zhiyuan1215.py")
        
        self.status_var = tk.StringVar(value="SYSTEM ONLINE // WAITING FOR INPUT")
        
        self.setup_ui()
        
        # Quick environment check
        threading.Thread(target=self.check_environment, daemon=True).start()

    def check_environment(self):
        """Check if the main script and python environment are accessible"""
        try:
            if os.path.exists(self.script_path):
                self.update_log(f"MAIN SCRIPT LOCATED: {os.path.basename(self.script_path)}")
                self.update_log(f"PYTHON INTERPRETER: {self.python_exe}")
                self.update_log("SYSTEM READY.")
            else:
                self.update_log(f"WARNING: Main script not found at {self.script_path}")
        except Exception as e:
            self.update_log(f"ENV CHECK ERROR: {str(e)}")

    def setup_ui(self):
        # --- Animated Background Canvas ---
        self.bg_canvas = tk.Canvas(self.root, width=800, height=600, bg=COLOR_BG, highlightthickness=0)
        self.bg_canvas.place(x=0, y=0, relwidth=1, relheight=1)
        
        # Create gradient effect with multiple rectangles
        for i in range(20):
            y = i * 30
            # Gradient from dark blue to lighter blue
            intensity = int(26 + i * 1.4)  # 0x1a to 0x3a
            color = f"#{intensity:02x}{intensity+5:02x}{39+i:02x}"
            self.bg_canvas.create_rectangle(0, y, 800, y+30, fill=color, outline="")
        
        # Add grid pattern
        for x in range(0, 800, 40):
            self.bg_canvas.create_line(x, 0, x, 600, fill=COLOR_ACCENT, width=1, stipple="gray12")
        for y in range(0, 600, 40):
            self.bg_canvas.create_line(0, y, 800, y, fill=COLOR_ACCENT, width=1, stipple="gray12")
        
        # Add some circuit-like decorative lines
        self.bg_canvas.create_line(50, 50, 200, 50, 200, 150, fill=COLOR_ACCENT_2, width=2, stipple="gray25")
        self.bg_canvas.create_line(700, 80, 550, 80, 550, 200, fill=COLOR_ACCENT_2, width=2, stipple="gray25")
        
        # Animated scan line
        self.scan_line = self.bg_canvas.create_line(0, 0, 800, 0, fill=COLOR_ACCENT, width=2)
        self.scan_y = 0
        self.animate_scan_line()
        
        # --- Header ---
        header_frame = tk.Frame(self.root, bg=COLOR_BG, pady=20)
        header_frame.place(relx=0.5, rely=0.08, anchor="center")
        
        # Main title with glow effect
        title_canvas = tk.Canvas(header_frame, width=700, height=60, bg=COLOR_BG, highlightthickness=0)
        title_canvas.pack()
        
        # Glow layers
        for offset in [4, 3, 2, 1]:
            alpha = 50 + (4-offset) * 30
            title_canvas.create_text(350, 25+offset, text="ROBOT CONTROL INTERFACE", 
                                     font=(FONT_FAMILY_HEADER, 26, "bold"), 
                                     fill=COLOR_ACCENT_2)
        
        # Main title text
        title_canvas.create_text(350, 25, text="ROBOT CONTROL INTERFACE", 
                                 font=(FONT_FAMILY_HEADER, 26, "bold"), 
                                 fill=COLOR_ACCENT)
        
        lbl_subtitle = tk.Label(header_frame, text="◢ ZHIYUAN LABS // NEURAL LINK TERMINAL v2.0.2 ◣", 
                                font=(FONT_FAMILY_MONO, 10), 
                                fg=COLOR_SUCCESS, bg=COLOR_BG)
        lbl_subtitle.pack()
    
        # --- Separator ---
        sep_canvas = tk.Canvas(self.root, width=760, height=3, bg=COLOR_BG, highlightthickness=0)
        sep_canvas.place(relx=0.5, rely=0.18, anchor="center")
        sep_canvas.create_rectangle(0, 0, 760, 1, fill=COLOR_ACCENT, outline="")
        sep_canvas.create_rectangle(0, 2, 760, 3, fill=COLOR_ACCENT_2, outline="")

        # --- Main Content Area ---
        content_frame = tk.Frame(self.root, bg=COLOR_BG, padx=40, pady=20)
        content_frame.place(relx=0.5, rely=0.45, anchor="center")

        # Left Side: Control Panel (Grid of buttons)
        label_instr = tk.Label(content_frame, text="◢ TARGET SELECTION MATRIX ◣",
                               font=(FONT_FAMILY_MONO, 13, "bold"),
                               fg=COLOR_ACCENT, bg=COLOR_BG, anchor="w")
        label_instr.grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 20))

        self.btn_frame = tk.Frame(content_frame, bg=COLOR_BG)
        self.btn_frame.grid(row=1, column=0, columnspan=4, sticky="nsew")

        # Create Grid of Buttons
        row = 0
        col = 0
        for item in ITEMS:
            btn = HighTechButton(self.btn_frame, text=item.upper(),
                                 command=lambda x=item: self.start_grasp_thread(x))
            btn.grid(row=row, column=col, padx=8, pady=8)
            col += 1
            if col > 3: # 4 buttons per row
                col = 0
                row += 1

        # --- Footer / Log ---
        log_frame = tk.Frame(self.root, bg=COLOR_BG_2, borderwidth=2, relief="flat")
        log_frame.place(relx=0.5, rely=0.85, anchor="center", width=760, height=180)
        
        # Decorative border for log with gradient
        log_border = tk.Canvas(log_frame, height=4, bg=COLOR_BG_2, highlightthickness=0)
        log_border.pack(fill="x", side="top")
        log_border.create_rectangle(0, 0, 760, 2, fill=COLOR_ACCENT, outline="")
        log_border.create_rectangle(0, 2, 760, 4, fill=COLOR_ACCENT_2, outline="")

        self.lbl_status = tk.Label(log_frame, textvariable=self.status_var,
                                   font=(FONT_FAMILY_MONO, 13, "bold"),
                                   fg=COLOR_SUCCESS, bg=COLOR_BG_2, pady=8)
        self.lbl_status.pack(anchor="w", padx=10)

        self.log_text = tk.Text(log_frame, height=5, bg=COLOR_BG, fg=COLOR_FG,
                                font=(FONT_FAMILY_MONO, 9), bd=0, state="disabled",
                                insertbackground=COLOR_ACCENT, selectbackground=COLOR_ACCENT_2)
        self.log_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    def animate_scan_line(self):
        """Animate a scanning line effect"""
        self.scan_y = (self.scan_y + 5) % 600
        self.bg_canvas.coords(self.scan_line, 0, self.scan_y, 800, self.scan_y)
        self.root.after(50, self.animate_scan_line)

    def update_log(self, message):
        """Standard log update with timestamp"""
        ts = time.strftime("%H:%M:%S")
        full_msg = f"[{ts}] {message}\n"
        
        def _update():
            self.log_text.config(state="normal")
            self.log_text.insert("end", full_msg)
            self.log_text.see("end")
            self.log_text.config(state="disabled")
        
        self.root.after(0, _update)

    def start_grasp_thread(self, label):
        if self.status_var.get().startswith("EXECUTING"):
            self.update_log("WARNING: OPERATION IN PROGRESS. WAIT.")
            return

        self.status_var.set(f"EXECUTING PROTOCOL: {label.upper()}...")
        self.update_log(f"COMMAND RECEIVED: GRASP {label.upper()}")
        
        thread = threading.Thread(target=self.execute_grasp, args=(label,))
        thread.start()

    def execute_grasp(self, label):
        try:
            # Call the main script via subprocess
            self.update_log(f"LAUNCHING GRASP PROCESS FOR {label.upper()}...")
            
            cmd = [self.python_exe, self.script_path, "--label", label]
            
            # Run the subprocess
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.path.dirname(self.script_path)
            )
            
            # Wait for completion
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                self.status_var.set("OPERATION SUCCESSFUL")
                self.update_log(f"GRASP {label.upper()} COMPLETE.")
                if stdout:
                    for line in stdout.strip().split('\n')[-3:]:  # Show last 3 lines
                        if line.strip():
                            self.update_log(f"  > {line.strip()}")
            else:
                self.status_var.set("OPERATION FAILED")
                self.update_log(f"GRASP {label.upper()} FAILED.")
                if stderr:
                    self.update_log(f"ERROR: {stderr.strip()[:100]}")
                
        except Exception as e:
            self.status_var.set("CRITICAL ERROR")
            self.update_log(f"EXECUTION ERROR: {str(e)}")
            print(e)
            
        # Reset status after delay
        time.sleep(2)
        if not self.status_var.get().startswith("CRITICAL"):
            self.status_var.set("SYSTEM READY")

if __name__ == "__main__":
    root = tk.Tk()
    app = RobotGUI(root)
    root.mainloop()
