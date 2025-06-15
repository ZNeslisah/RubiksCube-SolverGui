from tkinter import *
import cubie
import solver
import cv2
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
import numpy as np
import os
from vpython import box, vector, color, canvas

# === Model and Color Config ===
model_path = os.path.join(os.path.dirname(__file__), "color_classification_model2.h5")
model = load_model(model_path)

width = 90
facelet_id = [[[0 for _ in range(3)] for _ in range(3)] for _ in range(6)]
colorpick_id = [0 for _ in range(6)]
curcol = None
t = ("U", "R", "F", "D", "L", "B")
cols = ("yellow", "green", "red", "white", "blue", "orange")
label_names = ['blue', 'red', 'yellow', 'orange', 'white', 'green']
confirmed_faces = {}

color_name_to_index = {'white': 0, 'yellow': 1, 'blue': 2, 'green': 3, 'red': 4, 'orange': 5}

WINDOW_WIDTH = 12 * width + 20
WINDOW_HEIGHT = 9 * width + 20

main_root = Tk()
main_root.title("Rubik's Cube Solver")
main_root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
main_root.configure(bg="LightSkyBlue3")

current_prediction = None

# === VPython 3D Viewer ===
def show_3d_cube(face_colors):
    if len(face_colors) != 54:
        print("Invalid cube: need 54 facelet colors")
        return

    vpython_color_map = {
        'white': color.white,
        'yellow': color.yellow,
        'blue': color.blue,
        'green': color.green,
        'red': color.red,
        'orange': color.orange,
        'gray': color.gray(0.7)
    }

    win = canvas(title="3D Rubik's Cube", width=600, height=600, background=color.white)
    win.camera.pos = vector(5, 5, 10)
    win.camera.axis = vector(-5, -5, -10)

    face_offsets = {
        0: vector(0, 1, 0),
        1: vector(1, 0, 0),
        2: vector(0, 0, 1),
        3: vector(0, -1, 0),
        4: vector(-1, 0, 0),
        5: vector(0, 0, -1)
    }
    face_axes = {
        0: ('x', 'z'), 1: ('z', 'y'), 2: ('x', 'y'),
        3: ('x', 'z'), 4: ('z', 'y'), 5: ('x', 'y')
    }

    tile_size = 0.9
    gap = 0.05
    idx = 0
    for face in range(6):
        origin = face_offsets[face] * 3
        axis1, axis2 = face_axes[face]
        for i in range(3):
            for j in range(3):
                color_name = face_colors[idx] if idx < len(face_colors) else 'gray'
                cube_color = vpython_color_map.get(color_name, color.gray(0.5))
                pos = origin + vector(0, 0, 0)
                setattr(pos, axis1, getattr(pos, axis1) + (j - 1) * (tile_size + gap))
                setattr(pos, axis2, getattr(pos, axis2) - (i - 1) * (tile_size + gap))
                box(pos=pos, size=vector(0.85, 0.85, 0.1), color=cube_color)
                idx += 1

# === Helper ===
def flatten_faces():
    inverse_color_index = {v: k for k, v in color_name_to_index.items()}
    flat_colors = []
    for face_index in range(6):
        if face_index not in confirmed_faces:
            print(f"Face {face_index} missing")
            return None
        flat_colors += [inverse_color_index[i] for i in confirmed_faces[face_index]]
    return flat_colors

# === GUI Init ===
frame = Frame(main_root, width=WINDOW_WIDTH, height=WINDOW_HEIGHT, bg="white")
frame.pack(pady=30, expand=True)
frame.pack_propagate(False)

Label(frame, text="     Ms.Cube", bg="white", font=("Arial", 25, "bold")).pack(pady=20)
original_pil = Image.open("logo2.png")
resized = original_pil.resize((int(original_pil.width / 1.2), int(original_pil.height / 1.2)))
photo = ImageTk.PhotoImage(resized)

Label(frame, image=photo, bg="white", borderwidth=0, highlightthickness=0).pack(pady=5)
Label(frame, text="Select Input Method", bg="white", font=("Arial", 20)).pack(pady=10)

button_row = Frame(frame, bg="white")
button_row.pack(pady=20)

Button(button_row, text="Manual Input", font=("Arial", 18), bg="white", command=lambda: print("Manual Input"))\
    .pack(side=LEFT, padx=20)
Button(button_row, text="Scan Cube", font=("Arial", 18), bg="white", command=lambda: print("Scan Cube"))\
    .pack(side=LEFT, padx=20)
Button(frame, text="3D View", font=("Arial", 16), bg="white", command=lambda: show_3d_cube(flatten_faces()))\
    .pack(pady=10)

main_root.mainloop()
