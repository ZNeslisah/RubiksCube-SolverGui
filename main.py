from tkinter import *
import cubie
import solver  # or whatever your file is named, without the .py
import cv2
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
import numpy as np
import os
from tkinter import ttk
import re
from glob import glob
from sendToEsp import MotorCommand
import subprocess


def setup_bluetooth():
    try:
        subprocess.run(["sudo", "rfcomm", "release", "0"], check=True)
        subprocess.run(["sudo", "rfcomm", "bind", "0", "6C:C8:40:06:C4:F6", "1"], check=True)
        print("Bluetooth bound to /dev/rfcomm0")
    except subprocess.CalledProcessError as e:
        print(f"Bluetooth setup failed: {e}")

setup_bluetooth()

model_path = os.path.join(os.path.dirname(__file__), "color_classification_model3.h5")
model = load_model(model_path)

# Global variables
DEFAULT_HOST = 'localhost'
DEFAULT_PORT = '8080'
width = 90 #60  # facelet size
facelet_id = [[[0 for col in range(3)] for row in range(3)] for face in range(6)]
colorpick_id = [0 for i in range(6)]
curcol = None
t = ("U", "R", "F", "D", "L", "B")
cols = ("yellow", "green", "red", "white", "blue", "orange")

label_names = ['blue', 'red', 'yellow', 'orange', 'white', 'green']

pattern_sequences = {
    "Four Spots": "L2 R2 U1 D3 F2 B2 U1 D1",
    "Checkerboard": "R1 D1 L3 R3 D1 B2 U1 B1 U3 R1 D3 F1 R1 F1 D3 F1 B3 L1 U2 D1",  # example
    "Flipped Tips": "U1 R1 D3 L2 D1 R3 U3 F2 D1 L2 D3 F2 D1 L2 D3 F2",
    "Deckerboard": "U1 D1 F1 B3 L3 R1 U1 D3 F2 U1 F2 B2 D2 L2 R2 D1",
    "Vertical Stripes": "L1 U1 L1 F1 B2 R1 D3 F1 D2 B1 D3 R1 F2 B1 L1 U1 L1",
    "Python": "L2 F3 R3 U1 F3 B1 L3 B1 L3 R1 D3 F1 R1 B2",
    "Cube in cube": "L1 B1 L1 U3 F1 U1 L2 B2 U3 B3 R1 D3 R3 B2 U1",
    "Twister": "L1 F3 U1 B1 L3 B3 L1 U3 F1 U1 B3 U3 B1 L3",
    "Pong": "U1 F2 L2 R2 B2 U2 D1 F2 L2 R2 B2",
    "Scotish Skirt": "U3 F2 B2 L2 R2 U3 F1 B1 L1 R3 U1 L2 D2 F2 B2 L2 U2 L2 U3 L2"
}


confirmed_faces = {}  # key: face index (0-5), value: list of 9 int labels

selected_mode = None  # "pattern" or "solver"
selected_pattern = None

pattern_images = {}  # Holds loaded PhotoImages to prevent GC
pattern_image_label = None  # The label that shows the image


# Color mapping
color_name_to_index = {
    'white': 0,
    'yellow': 1,
    'blue': 2,
    'green': 3,
    'red': 4,
    'orange': 5
}


# Constants for window size
WINDOW_WIDTH = 12 * width + 20
WINDOW_HEIGHT = 9 * width + 20

# Main window
main_root = Tk()
main_root.title("Rubik's Cube Solver")
main_root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
main_root.configure(bg="LightSkyBlue3")

current_prediction = None

def back_from_input_select():
    input_select_frame.place_forget()
    if selected_mode == "Create Pattern":
        pattern_select_frame.place(x=0,y=0)
    else:
        mode_select_frame.place(relx=0.5, rely=0.5, anchor="center")

def back_from_pattern_select():
    pattern_select_frame.place_forget()
    mode_select_frame.place(relx=0.5, rely=0.5, anchor="center")


# Transition function to manual input window
def open_manual_input():
    main_root.withdraw()
    manual_input_window()

def open_camera_window():
    global current_prediction
    full_solution = None


    last_frame = None  # Holds the most recent camera frame
    box_w = 90
    box_h = 90
    width_cam=int(640)
    height_cam=int(480)
    start_x = (width_cam - 3 * box_w) // 2
    start_y = (height_cam - 3 * box_h) // 2
    
    current_prediction = None  # will hold the latest prediction as 9 labels
    


    def capture_boxes():
        global current_prediction
        if last_frame is None:
            # print("No frame to capture")
            return

        facelets = []
        facelet_coords = []

        for i in range(3):
            for j in range(3):
                x1 = start_x + (2 - j) * box_w  # <- REVERSED J
                y1 = start_y + i * box_h
                x2 = x1 + box_w
                y2 = y1 + box_h

                cropped = last_frame[y1:y2, x1:x2]
                resized = cv2.resize(cropped, (32, 32))  # match model input
                normalized = resized.astype('float32') / 255.0
                facelets.append(normalized)
                facelet_coords.append((i, j, cropped))

        # Predict
        facelets_np = np.array(facelets)
        predictions = model.predict(facelets_np)
        predicted_indices = np.argmax(predictions, axis=1)
        predicted_labels = [label_names[idx] for idx in predicted_indices]

        # print("Predicted Colors:")

        # Draw color face preview on the canvas
        color_preview_canvas.delete("all")

        facelet_size = 50

        for i in range(3):
            for j in range(3):
                label = predicted_labels[i * 3 + j]
                color_preview_canvas.create_rectangle(
                    (2 - j) * facelet_size, i * facelet_size,  # flip j: (2 - j)
                    (3 - j) * facelet_size, (i + 1) * facelet_size,
                    fill=label, outline="black"
                )


        current_prediction = predicted_labels
    # Hide Capture button
            
        btn_capture.grid_forget()

        btn_tick.grid(row=0, column=0, padx=10)
        btn_cross.grid(row=0, column=1, padx=10)


        # # Save cropped images with predicted color names
        # output_dir = "classified_facelets"
        # os.makedirs(output_dir, exist_ok=True)

        # for idx, (i, j, img) in enumerate(facelet_coords):
        #     label = predicted_labels[idx]
        #     filename = f"{label}_{i}_{j}.png"
        #     path = os.path.join(output_dir, filename)
        #     cv2.imwrite(path, img)
        output_dir = "classified_facelets"
        os.makedirs(output_dir, exist_ok=True)

        # Get next available image index
        existing = glob(os.path.join(output_dir, "*.png"))
        next_id = len(existing)

        for idx, (i, j, img) in enumerate(facelet_coords):
            label = predicted_labels[idx]
            filename = f"{label}_{i}_{j}_{next_id + idx}.png"
            path = os.path.join(output_dir, filename)
            cv2.imwrite(path, img)
        

    def draw_unfolded_face(face_idx):
        face_positions = {
            0: (3, 0),  # U (yellow)
            3: (3, 6),  # D (white)
            4: (0, 3),  # L (blue)
            2: (3, 3),  # F (red)
            1: (6, 3),  # R (green)
            5: (9, 3)   # B (orange)
        }

        index_to_color = {v: k for k, v in color_name_to_index.items()}
        if face_idx not in confirmed_faces:
            return

        face = confirmed_faces[face_idx]
        x_off, y_off = face_positions[face_idx]
        tile = 25
        for i in range(3):
            for j in range(3):
                color_index = face[i * 3 + j]
                color = index_to_color[color_index]
                x = (x_off + j) * tile
                y = (y_off + i) * tile
                cube_preview_canvas.create_rectangle(x, y, x + tile, y + tile, fill=color, outline="black")


    color_to_face_index = {
    'yellow': 0,  # U
    'white': 3,   # D
    'blue': 4,    # L
    'red': 2,     # F
    'green': 1,   # R
    'orange': 5   # B
    }

    


    def save_prediction():
        global current_prediction
        global full_solution
        if current_prediction:
            int_labels = [color_name_to_index[label] for label in current_prediction]

            # Identify face index from center color
            center_color = current_prediction[4]
            # print(f'center color: {center_color}')
            if center_color not in color_to_face_index:
                # print("Unknown center color:", center_color)
                return

            face_index = color_to_face_index[center_color]
            # print(f'face index {face_index}')
            confirmed_faces[face_index] = int_labels
            # print(f'confirmed faces {confirmed_faces}')

            # print(f"Saved face {face_index} ({center_color}):", int_labels)
            draw_unfolded_face(face_index)

            num_saved = len(confirmed_faces)
            if num_saved < 6:
                info_label.config(text=capture_instructions[num_saved])
            elif num_saved == 6:
                info_label.config(text="✅ All 6 faces captured! You can now click Solve.")

            btn_tick.grid_forget()
            btn_cross.grid_forget()

            btn_capture.grid(row=0, column=0, columnspan=2, padx=20)


        else:
            pass
            # print("No prediction to save")

        

    def discard_prediction():
        global current_prediction
        # print("Prediction discarded.")
        current_prediction = None

        btn_tick.grid_forget()
        btn_cross.grid_forget()
        # Show Capture button again
        btn_capture.grid(row=0, column=0, columnspan=2, padx=20)


    def build_solver_string():
        if len(confirmed_faces) != 6:
            # print("Error: Need exactly 6 faces scanned.")
            return None

        face_char_order = ["U", "R", "F", "D", "L", "B"]  # expected order
        center_color_to_face = {}

        for face_index in range(6):
            if face_index not in confirmed_faces:
                # print(f"Missing face {face_index}")
                return None

            face = confirmed_faces[face_index]
            center_color_index = face[4]
            center_color_to_face[center_color_index] = face_char_order[face_index]

        # Build definition string
        definition = ""
        for face_index in range(6):
            face = confirmed_faces[face_index]
            definition += ''.join(center_color_to_face[color] for color in face)

        # print(f'Cube state: {definition}')
        return definition
    
    def start_blink():
        current = blinking_label.cget("text")
        blinking_label.config(text="" if current else "Position the cube correctly, then press the start button!")
        blinking_label.after(1000, start_blink)

    def send():
        global full_solution
        data = MotorCommand(full_solution)
        data.convert_to_sequence()
        data.generate_motor_commands()
        data.send_sequence()

    def solve_cube():
        nonlocal solution_box, blinking_label
        global full_solution

        definition_string = build_solver_string()
        if definition_string is None:
            return

        try:
            solution = solver.solve(definition_string)
            solution_clean = re.sub(r"\(\d+f\)", "", solution).strip()
            # If "Create Pattern" mode, append pattern sequence
            if selected_mode == "Create Pattern":
                if selected_pattern in pattern_sequences:
                    pattern_seq = pattern_sequences[selected_pattern]
                    pattern_clean = pattern_seq.strip()
                    full_solution = solution_clean + " " + pattern_clean
                else:
                    # print("No valid pattern selected")
                    full_solution = solution_clean
            else:
                full_solution = solution_clean

            # print("Solution:", full_solution)

            solution_box.delete(1.0, END)
            solution_box.insert(INSERT, f"Solution:\n{full_solution}")

            # data = MotorCommand(full_solution)
            # data.convert_to_sequence()
            # data.generate_motor_commands()

            # blinking_label.config(text="Load the cube to the machine!")
            # start_blink()

            def startMachineProcess():            
                blinking_label.config(text="Position the cube correctly, then press the start button")          
                # canvas.create_window(175 + 6.5 * width, 35 + 1.8 * width, anchor=NW, window=bsend)
                # bsend.place(x=310, y=720)
                bsend.place(x=310, y=700) 

            frame.after(2000, startMachineProcess)




        except Exception as e:
            solution_box.delete(1.0, END)
            solution_box.insert(INSERT, f"Error:\n{str(e)}")



    def initialize_cube_preview():

        face_positions = {
            0: (3, 0),  # yellow (U)
            3: (3, 6),  # white (D)
            4: (0, 3),  # blue  (L)
            2: (3, 3),  # red   (F)
            1: (6, 3),  # green (R)
            5: (9, 3)   # orange(B)
        }

        face_to_color = {
            0: "yellow", 1: "green", 2: "red",
            3: "white", 4: "blue", 5: "orange"
        }
        size = 25
        for face_idx, (x_off, y_off) in face_positions.items():
            for i in range(3):
                for j in range(3):
                    x = (x_off + j) * size
                    y = (y_off + i) * size
                    fill = face_to_color[face_idx] if (i, j) == (1, 1) else "gray90"
                    cube_preview_canvas.create_rectangle(x, y, x + size, y + size, fill=fill, outline="black")


    for widget in frame.winfo_children():
        widget.place_forget()

    frame.configure(bg="white")

    def back_from_camera_view():
        # Stop the camera
        cap.release()

        # Clear the current camera UI
        for widget in frame.winfo_children():
            widget.place_forget()

        # Show the previous screen
        input_select_frame.place(x=0, y=0)


    camera_frame = Frame(frame, bg="white")
    camera_frame.place(x=50, y=50)
 
    color_preview_canvas = Canvas(camera_frame, width=180, height=180, bg="white", highlightthickness=0)
    color_preview_canvas.grid(row=1, column=1, pady=(20,0))
    color_preview_canvas.place(x=770, y=210)  # Adjust x, y as needed

    cube_preview_canvas = Canvas(frame, width=560, height=370, bg="white", highlightthickness=0)
    cube_preview_canvas.place(x=730, y=500)  # Or another y that doesn’t overlap anything
    initialize_cube_preview()


    cam_canvas = Canvas(camera_frame, width=width_cam, height=height_cam, bg="black", highlightthickness=0)
    cam_canvas.grid(row=0, column=0)

    button_panel = Frame(camera_frame, bg="white")
    button_panel.grid(row=0, column=1, padx=80, pady=0, sticky="n")

    capture_instructions = [
    "Face red toward the camera. Blue should be on the right.",
    "Rotate left to blue face.",
    "Rotate left to orange face.",
    "Rotate left to green face.",
    "Rotate left to red again. Tilt up to show yellow.",
    "Back to red. Tilt down to show white."
    ]


    info_label = Label(button_panel, text=capture_instructions[0], font=("Arial", 15),
                   bg="white", wraplength=200, justify="left", height=4)
    info_label.pack(padx=35, pady=(20, 0))


    button_slot = Frame(frame, bg="white", width=250, height=80)
    button_slot.pack_propagate(False)  # Prevents resizing based on inner widgets
    button_slot.place(x=800, y=200)

    # Capture button (centered in row 0, column
    #  0)
    btn_capture = Button(button_slot, text="Capture", font=("Arial", 16), bg="white", command=capture_boxes)
    btn_capture.grid(row=0, column=0, columnspan=2, padx=20)

    # Tick and Cross — don't show yet
    btn_tick = Button(button_slot, text="✓", font=("Arial", 16), width=4, fg="green", bg="white", command=save_prediction)
    btn_cross = Button(button_slot, text="✗", font=("Arial", 16), width=4, fg="red", bg="white", command=discard_prediction)
    # Don't .grid() them yet
   
    btn_solve = Button(frame, text="Solve", font=("Arial", 18), bg="white", command=solve_cube)
    # btn_solve.grid(row=1, column=0, pady=(10, 0))
    btn_solve.place(x=90, y=580)

    # Text box to show the solution
    solution_box = Text(frame, height=4, width=40, font=("Arial", 12))
    solution_box.place(x=200, y= 560)

    # Blinking label
    blinking_label = Label(frame, text="", font=("Arial", 14, "bold"), fg="red", bg="white")
    blinking_label.place(x=160, y=650)

    bsend = Button(frame,text="Start", height=2, width=10, command= send)
    

    back_btn_camera = Button(frame, text="← Back", font=("Arial", 12), command=back_from_camera_view)
    back_btn_camera.place(x=10, y=10)



    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        # print("❌ Failed to open camera")
        return  # Or show error popup

    def show_frame():
        ret, frame_cv = cap.read()
        if ret:
            frame_cv = cv2.flip(frame_cv, 1)
            frame_cv = cv2.resize(frame_cv, (width_cam, height_cam))
            cv_img = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv_img)
            imgtk = ImageTk.PhotoImage(image=img)
            cam_canvas.imgtk = imgtk
            cam_canvas.create_image(0, 0, anchor=NW, image=imgtk)

            cam_canvas.delete("overlay")
            for i in range(3):
                for j in range(3):
                    x1 = start_x + j * box_w
                    y1 = start_y + i * box_h
                    x2 = x1 + box_w
                    y2 = y1 + box_h
                    cam_canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2, tags="overlay")

        nonlocal last_frame
        last_frame = frame_cv.copy()

        cam_canvas.after(10, show_frame)

    show_frame()

    
def manual_input_window():
    global root, canvas, display, txt_host, txt_port
    full_solution = None

    root = Toplevel(main_root)
    root.grab_set()  # Optional: Focus input on this window

    def go_back_to_main():
        root.destroy()
        main_root.deiconify()  # Show the main window again

    root.protocol("WM_DELETE_WINDOW", main_root.quit)
    root.wm_title("Solver Client")
    root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
    canvas = Canvas(root, width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
    canvas.pack()

    # Output text boxes
    display_solution = Text(root,height=4, width=55)
    canvas.create_window(20 + 6.5 * width, 20 + 0.5 * width, anchor=NW, window=display_solution)


    blinking_label = Label(root, text="", font=("Arial", 14, "bold"), fg="red", bg=root["bg"])
    blinking_label.place(x=15 + 6.5 * width, y=1.8 * width)

    def show_text(txt):
        # print(txt)
        display.insert(INSERT, txt)
        root.update_idletasks()

    def create_facelet_rects(a):
        offset = ((1, 0), (2, 1), (1, 1), (1, 2), (0, 1), (3, 1))
        for f in range(6):
            for row in range(3):
                y = 10 + offset[f][1] * 3 * a + row * a
                for col in range(3):
                    x = 10 + offset[f][0] * 3 * a + col * a
                    facelet_id[f][row][col] = canvas.create_rectangle(x, y, x + a, y + a, fill="grey")
                    if row == 1 and col == 1:
                        canvas.create_text(x + width // 2, y + width // 2, font=("", 14), text=t[f], state=DISABLED)
        for f in range(6):
            canvas.itemconfig(facelet_id[f][1][1], fill=cols[f])

    def create_colorpick_rects(a):
        global curcol
        for i in range(6):
            x = (i % 3)*(a+5) + 7*a
            y = (i // 3)*(a+5) + 7*a
            colorpick_id[i] = canvas.create_rectangle(x, y, x + a, y + a, fill=cols[i])
        canvas.itemconfig(colorpick_id[0], width=4)
        curcol = cols[0]

    def get_definition_string():
        color_to_facelet = {canvas.itemcget(facelet_id[i][1][1], "fill"): t[i] for i in range(6)}
        return ''.join(color_to_facelet[canvas.itemcget(facelet_id[f][r][c], "fill")] for f in range(6) for r in range(3) for c in range(3))

    def solve():
        global full_solution
        display_solution.delete(1.0, END)
        try:
            defstr = get_definition_string()
        except Exception as e:
            display_solution.insert(INSERT, f"Invalid cube: {str(e)}\n")
            return

        root.update_idletasks()
        # print(f'cube state: {defstr}')

        try:
            solution = solver.solve(defstr)
            solution_clean = re.sub(r"\(\d+f\)", "", solution).strip()

            # Handle pattern addition
            if selected_mode == "Create Pattern":
                if selected_pattern in pattern_sequences:

                    pattern_seq = pattern_sequences[selected_pattern]
                    pattern_clean = pattern_seq.strip()
                    full_solution = solution_clean + " " + pattern_clean

                else:
                    # print("No valid pattern selected")
                    full_solution = solution_clean
            else:
                full_solution = solution_clean

            display_solution.insert(INSERT, f"Solution:\n{full_solution}")


            # def start_blink():
            #     current = blinking_label.cget("text")
            #     blinking_label.config(text="" if current else "Position the cube correctly, then press the start button")
            #     root.after(1500, start_blink)

            # blinking_label.config(text="Position the cube correctly, then press the start button")
            # start_blink()

            # # Show the Start button after 2 seconds

            # def show_start_button():
            #     canvas.create_window(175 + 6.5 * width, 35 + 1.8 * width, anchor=NW, window=bsend)

            # root.after(4000, show_start_button)

            def startMachineProcess():            
                blinking_label.config(text="Position the cube correctly, then press the start button")          
                canvas.create_window(175 + 6.5 * width, 35 + 1.8 * width, anchor=NW, window=bsend)
            root.after(2000, startMachineProcess)
           

        except Exception as e:
            display.insert(INSERT, f"Error while solving:\n{str(e)}\n")

    def send():
        global full_solution
        data = MotorCommand(full_solution)
        # print(full_solution)
        data.convert_to_sequence()
        data.generate_motor_commands()
        data.send_sequence()

    def clean():
        for f in range(6):
            for row in range(3):
                for col in range(3):
                    canvas.itemconfig(facelet_id[f][row][col], fill=canvas.itemcget(facelet_id[f][1][1], "fill"))

    def empty():
        for f in range(6):
            for row in range(3):
                for col in range(3):
                    if (row, col) != (1, 1):
                        canvas.itemconfig(facelet_id[f][row][col], fill="grey")

    def random():
        cc = cubie.CubieCube()
        cc.randomize()
        fc = cc.to_facelet_cube()
        idx = 0
        for f in range(6):
            for row in range(3):
                for col in range(3):
                    canvas.itemconfig(facelet_id[f][row][col], fill=cols[fc.f[idx]])
                    idx += 1

    def click(_):
        global curcol
        idlist = canvas.find_withtag("current")
        if idlist:
            if idlist[0] in colorpick_id:
                curcol = canvas.itemcget("current", "fill")
                for i in range(6):
                    canvas.itemconfig(colorpick_id[i], width=1)
                canvas.itemconfig("current", width=5)
            else:
                canvas.itemconfig("current", fill=curcol)

    bsolve = Button(root, text="Solve", height=2, width=10, command=solve)
    bsend = Button(root, text="Start", height=2, width=10, command=send )
    back_btn = Button(root, text="← Back", font=("Arial", 12), command=go_back_to_main)
    bclean = Button(root, text="Clean", width=10, command=clean)
    bempty = Button(root, text="Empty", width=10, command=empty)
    brandom = Button(root, text="Random", width=10, command=random)
    back_btn.place(x=10, y=10)

    canvas.create_window(10 + 10.5 * width, 10 + 6.5 * width, anchor=NW, window=bsolve)
    canvas.create_window(10 + 10.5 * width, 10 + 7.5 * width, anchor=NW, window=bclean)
    canvas.create_window(10 + 10.5 * width, 10 + 8 * width, anchor=NW, window=bempty)
    canvas.create_window(10 + 10.5 * width, 10 + 8.5 * width, anchor=NW, window=brandom)
    txt_host = Text(height=1, width=20)
    txt_host.insert(INSERT, DEFAULT_HOST)
    txt_port = Text(height=1, width=20)
    txt_port.insert(INSERT, DEFAULT_PORT)

    canvas.bind("<Button-1>", click)
    create_facelet_rects(width)
    create_colorpick_rects(width)
    root.mainloop()


# --- Main frame (shared) ---
frame = Frame(main_root, width=WINDOW_WIDTH, height=WINDOW_HEIGHT, bg="white")
frame.pack(pady=30, expand=True)
frame.pack_propagate(False)

# Load and store logo once
original_pil = Image.open("logo2.png")
resized = original_pil.resize((int(original_pil.width / 1.2), int(original_pil.height / 1.2)))
photo = ImageTk.PhotoImage(resized)

# --- Subframe 1: Solver Mode Selection ---
mode_select_frame = Frame(frame, bg="white")
mode_select_frame.place(relx=0.5, rely=0.5, anchor="center")

Label(mode_select_frame, text="     RoboCube", bg="white", font=("Arial", 28, "bold")).pack(pady=20)
Label(mode_select_frame, image=photo, bg="white").pack(pady=5)
Label(mode_select_frame, text="Select Solver Mode", bg="white", font=("Arial", 20)).pack(pady=10)

mode_button_row = Frame(mode_select_frame, bg="white")
mode_button_row.pack(pady=20)

def select_mode(mode):
    global selected_mode
    selected_mode = mode
    mode_select_frame.place_forget()

    if mode == "Create Pattern":
        pattern_select_frame.place(x=0, y=0)
    else:
        input_select_frame.place(x=0, y=0)


Button(mode_button_row, text="Create Pattern", font=("Arial", 18), bg="white", command=lambda: select_mode("Create Pattern")).pack(side=LEFT, padx=20)
Button(mode_button_row, text="Solve Cube", font=("Arial", 18), bg="white", command=lambda: select_mode("Solve Cube")).pack(side=LEFT, padx=20)

# --- Subframe 2: Patter Method Selection ---
pattern_select_frame = Frame(frame, bg="white")

Label(pattern_select_frame, text="     RoboCube", bg="white", font=("Arial", 28, "bold")).pack(pady=(55,24))
Label(pattern_select_frame, image=photo, bg="white").pack(padx =218.5)
Label(pattern_select_frame, text="Select a Pattern", bg="white", font=("Arial", 20)).pack(pady=10)

back_btn_pattern = Button(pattern_select_frame, text="← Back", font=("Arial", 12), command=back_from_pattern_select)
back_btn_pattern.place(x=10, y=10)

pattern_var = StringVar()
pattern_combo = ttk.Combobox(pattern_select_frame, textvariable=pattern_var, font=("Arial", 14), width=30)
pattern_combo['values'] = list(pattern_sequences.keys())
pattern_combo.pack(pady=10)

pattern_image_label = Label(frame, bg="white")
pattern_image_label.place(x = 750, y = 550)

def update_pattern_image(event=None):
    selected = pattern_var.get()
    image_path = os.path.join("patterns", selected + ".png")
    try:
        img = Image.open(image_path)
        img = img.resize((200, 200))  # Resize as needed
        tk_img = ImageTk.PhotoImage(img)
        pattern_images[selected] = tk_img  # Prevent garbage collection
        pattern_image_label.config(image=tk_img)
    except Exception as e:
        pattern_image_label.config(image='', text="No image available")
        # print(f"Image not found for pattern '{selected}':", e)

# Bind dropdown change to image update
pattern_combo.bind("<<ComboboxSelected>>", update_pattern_image)


def proceed_after_pattern():
    global selected_pattern
    selected_pattern = pattern_var.get()
    if not selected_pattern:
        # print("No pattern selected")
        return
    # print("Selected pattern:", selected_pattern)
    pattern_image_label.config(image="", text="")
    pattern_select_frame.place_forget()
    input_select_frame.place(x=0,y=0)

Button(pattern_select_frame, text="Next", font=("Arial", 16), command=proceed_after_pattern).pack(pady=10)

# --- Subframe 3: Input Method Selection ---
input_select_frame = Frame(frame, bg="white")


Label(input_select_frame, text="     RoboCube", bg="white", font=("Arial", 28, "bold")).pack(pady=(55,24))
Label(input_select_frame, image=photo, bg="white").pack(padx = 218.5)
Label(input_select_frame, text="Select Input Method", bg="white", font=("Arial", 20)).pack(pady=15)

input_button_row = Frame(input_select_frame, bg="white")
input_button_row.pack(pady=18)

back_btn_input = Button(input_select_frame, text="← Back", font=("Arial", 12), command=back_from_input_select)
back_btn_input.place(x=10, y=10)
Button(input_button_row, text="Manual Input", font=("Arial", 18), bg="white", command=open_manual_input).pack(side=LEFT, padx=20)
Button(input_button_row, text="Scan Cube", font=("Arial", 18), bg="white", command=open_camera_window).pack(side=LEFT, padx=20)


main_root.mainloop()