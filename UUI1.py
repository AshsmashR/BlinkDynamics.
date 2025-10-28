# type : ignore

import mediapipe as mp
from PyQt5 import QtWidgets, QtGui, QtCore
import sys, csv, os, time
import numpy as np
from PyQt5.QtWidgets import QLabel, QHBoxLayout
from PyQt5.QtGui import QPixmap
import embeddings3_rcl
# Resource file for images and gifs
# ---------- External modules for detection ----------
from openpyxl import Workbook, load_workbook
from datetime import datetime
import os

import cv2
from collections import deque
from BLDT import BlinkAlgo


import csv
import os
from datetime import datetime
import sys
import os

def resource_path(rel_path):
    if getattr(sys, 'frozen', False):
        # Running as compiled .exe
        base_path = os.path.dirname(sys.executable)
    else:
        # Running as script
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, rel_path)

CSV_FILE = resource_path(os.path.join("UIUI", "userreg.csv"))
BASELINE_CSV = resource_path(os.path.join("UIUI", "baseline_results.csv"))
AFTERBASELINE_CSV = resource_path(os.path.join("UIUI", "afterbaseline_results.csv"))
RANDOM_CSV = resource_path(os.path.join("UIUI", "random_results.csv"))

LOGO_PATH = resource_path(os.path.join("UIUI", "my_logo.ico"))

# User registry fields
CSV_FIELDS = ["serial_no", "name", "ID", "baseline_per_min"]

# Headers for CSV test files
BASELINE_HEADERS = [
    'name', 'id', 'baseline1', 'baseline2', 'baseline3', 'baseline4', 'baseline5', 'baseline_avg',
    'mode', 'total_blinks', 'blink_rate_per_min', 'avg_blink_duration', 'perclos_p80', 'status', 'test_start_time'
]

OTHER_HEADERS = [
    'name', 'id', 'baseline_ref', 'mode', 'total_blinks',
    'blink_rate_per_min', 'avg_blink_duration', 'perclos_p80', 'status', 'test_start_time'
]

def ensure_file_exists(path, headers):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

def read_all_users():
    ensure_file_exists(CSV_FILE, CSV_FIELDS)
    with open(CSV_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)

def save_all_users(users):
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(users)

def id_exists(uid):
    return any(u["ID"].strip() == uid.strip() for u in read_all_users())

def get_next_serial():
    users = read_all_users()
    serials = [int(u.get("serial_no", 0) or 0) for u in users]
    return max(serials, default=0) + 1

def append_new_user(name, uid):
    if not name or not uid or id_exists(uid):
        return False
    users = read_all_users()
    serial_no = get_next_serial()
    new_user = {"serial_no": serial_no, "name": name, "ID": uid, "baseline_per_min": ""}
    users.append(new_user)
    save_all_users(users)
    print(f"[+] Added user {name} with ID {uid}")
    return True

def append_baseline_to_registry(user_id, baseline_bpm):
    if baseline_bpm is None:
        return
    users = read_all_users()
    updated = False
    for u in users:
        if u["ID"].strip() == user_id.strip():
            current = u.get("baseline_per_min", "")
            vals = [v.strip() for v in current.split(",") if v.strip()]
            vals.append(str(round(float(baseline_bpm), 2)))
            vals = vals[-10:]  # Keep last 10
            u["baseline_per_min"] = ",".join(vals)
            updated = True
            break
    if updated:
        save_all_users(users)

def get_latest_baseline_avg(user_id):
    users = read_all_users()
    for u in users:
        if u["ID"].strip() == user_id.strip():
            bstr = u.get("baseline_per_min", "")
            if bstr:
                vals = [float(v) for v in bstr.split(",") if v]
                if vals:
                    return round(sum(vals[-5:]) / len(vals[-5:]), 2)
    return None

def save_test_result(user_dict, test_result, mode, test_start_time=None):
    name = user_dict.get("name", "")
    user_id = user_dict.get("id", "")
    cpr = test_result.get("checkpoint_results", {}).get(145, {})

    if mode == "baseline":
        ensure_file_exists(BASELINE_CSV, BASELINE_HEADERS)
        prev_baselines = []
        if os.path.exists(BASELINE_CSV):
            with open(BASELINE_CSV, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("id", "").strip() == user_id.strip():
                        for i in range(1, 6):
                            v = row.get(f"baseline{i}", "")
                            if v:
                                try: prev_baselines.append(float(v))
                                except: pass
        baseline_bpm = test_result.get("personal_baseline_bpm", None)
        if baseline_bpm is not None:
            prev_baselines.append(float(baseline_bpm))
        prev_baselines = prev_baselines[-5:]
        bs = prev_baselines + [""] * (5 - len(prev_baselines))
        avg = round(sum(bs_i for bs_i in bs if bs_i != "") / len([bs_i for bs_i in bs if bs_i != ""]), 2) if any(bs) else ""
        row = [
            name, user_id,
            bs[0], bs[1], bs[2], bs[3], bs[4],
            avg,
            "baseline",
            test_result.get("total_blinks", ""),
            cpr.get("blink_rate", {}).get("blink_rate_per_min", ""),
            cpr.get("blink_duration", {}).get("avg_duration", ""),
            cpr.get("perclos", {}).get("perclos_p80", ""),
            cpr.get("classification_blink_rate", {}).get("status", ""),
            test_start_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ]
        with open(BASELINE_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)
        # Append baseline to registry CSV also for user
        if baseline_bpm is not None:
            append_baseline_to_registry(user_id, baseline_bpm)
        print(f"Saved baseline result for user {user_id}")

    elif mode == "after_baseline":
        ensure_file_exists(AFTERBASELINE_CSV, OTHER_HEADERS)
        baseline_ref = get_latest_baseline_avg(user_id)
        row = [
            name, user_id, baseline_ref, "after_baseline",
            test_result.get("total_blinks", ""),
            cpr.get("blink_rate", {}).get("blink_rate_per_min", ""),
            cpr.get("blink_duration", {}).get("avg_duration", ""),
            cpr.get("perclos", {}).get("perclos_p80", ""),
            cpr.get("classification_blink_rate", {}).get("status", ""),
            test_start_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ]
        with open(AFTERBASELINE_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)
        print(f"Saved after-baseline result for user {user_id}")

    else:  # random test
        ensure_file_exists(RANDOM_CSV, OTHER_HEADERS)
        baseline_ref = get_latest_baseline_avg(user_id)
        row = [
            name, user_id, baseline_ref, "random",
            test_result.get("total_blinks", ""),
            cpr.get("blink_rate", {}).get("blink_rate_per_min", ""),
            cpr.get("blink_duration", {}).get("avg_duration", ""),
            cpr.get("perclos", {}).get("perclos_p80", ""),
            cpr.get("classification_blink_rate", {}).get("status", ""),
            test_start_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ]
        with open(RANDOM_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)
        print(f"Saved random test result for user {user_id}")


#===============Drowsiness class ==========#
max_seconds=175
class BlinkAccessor(QtCore.QObject):
    """
    Runs the blink/drowsiness detection loop in a worker thread
    and emits signals with final results to the UI.
    """
    finished = QtCore.pyqtSignal(dict)
    error = QtCore.pyqtSignal(str)

    def __init__(self, mode="baseline", user_baseline=None, parent=None):
        super().__init__(parent)
        self.mode = mode  # "baseline" or "testing"
        self.user_baseline = user_baseline
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            # Create BlinkAlgo instance and run it
            print(f"[DEBUG] Starting BlinkAlgo with mode = {self.mode}")

            detector = BlinkAlgo(max_seconds=175, show_window=True, mode=self.mode)
            if self.mode == "after_baseline" and self.user_baseline is not None:
                detector.personal_baseline_blinks_per_min = self.user_baseline
            
            result = detector.run()

            if "error" in result:
                self.error.emit(result["error"])
            else:
                self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

# ============== MAIN APP UI ==============
class WelcomeApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drowsiness detection test")
        self.setGeometry(100, 100, 800, 800)
        self.setFixedSize(800, 800)
        self.current_widget = None
        self.user_type = None
        try:
            self.setWindowIcon(QtGui.QIcon(r":/images/csir.png"))
        except:
            pass

        self.setStyleSheet("""
    QMainWindow { background-color: #f0f0f0; }
    QLabel { color: #000000; }
    QLabel[role="title"] {
        color: #000000;
        background-color: transparent;
        padding: 10px;
    }
    QLabel[role="panel"] {
        color: #000000;
        background-color: transparent;
        padding: 10px;
    }
    QPushButton {
        background-color: #00008B;  /* dark blue */
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        font-weight: bold;
        min-width: 100px;       /* Specify a smaller min width */
        max-width: 200px;
    }
    QPushButton:hover { background-color: #154360; }
    QLineEdit {
        background-color: #D6EAF8;  /* light blue */
        color: #154360;            /* dark blue */
        padding: 6px 10px;
        border: 1px solid #154360;  /* dark blue */
        border-radius: 4px;
    }
""")
        
        

        self.mode = None  # "baseline" or "testing"
        self.thread = None
        self.worker = None

        self.user_selected = False  # Track selection state for page 2 "Next" button
        self.user_type = None  # Store new/existing user choice
        

        self.show_welcome_page()
    
    def get_baseline_list(self, user_id):
        baseline_values = []
        rows = read_all_users()  # Uses your CSV reading function
        for row in rows:
             if (row.get("ID") or "").strip() == user_id:
                baseline_str = row.get("baseline_per_min", "") or row.get("personal_baseline_bpm", "")
                if baseline_str:
                   try:
                       values = [float(x.strip()) for x in baseline_str.split(",") if x.strip()]
                       baseline_values.extend(values)
                   except:
                        try:
                           baseline_values.append(float(baseline_str))
                        except:
                          pass
        return baseline_values[-5:] if len(baseline_values) >= 5 else baseline_values
    
    
    def _update_page(self, widget):
        """Replace the central widget with the given widget."""
        if self.current_widget:
            self.current_widget.setParent(None)
        self.setCentralWidget(widget)
        self.current_widget = widget

    # ---------------- PAGE 1 ----------------
    def show_welcome_page(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setAlignment(QtCore.Qt.AlignCenter)

        # Background
        try:
            movie = QtGui.QMovie(r":\gifs\1smooth_loop.gif")  # path to your fade transition GIF
            background = QtWidgets.QLabel(widget)
            background.setMovie(movie)
            background.setGeometry(0, 0, 800, 800)
            movie.start()
            background.lower()
        except:
            widget.setStyleSheet("background-color: #f0f0f0;")

        logos_layout = QHBoxLayout()
        logos_layout.setSpacing(50)
        logo_paths = [
            r":/images/csir.png",
            r":/images/Central_Electronics_Engineering_Research_Institute_Logo.png",
            r":/images/csio.jpg",
            r":/images/Ias.jpeg",
            r":/images/IGIB_LOGO.png"
        ]

        for path in logo_paths:
            try:
                pixmap = QPixmap(path).scaled(90, 90, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
                label = QLabel()
                label.setPixmap(pixmap)
                logos_layout.addWidget(label)
            except Exception as e:
                print(f"Error loading logo {path}: {e}")

        layout.addLayout(logos_layout)
        layout.addSpacerItem(QtWidgets.QSpacerItem(20, 20))

        # Logo/GIF
        try:
            movie = QtGui.QMovie(r":/gifs/eyevideo.gif")
            gif_label = QtWidgets.QLabel()
            gif_label.setMovie(movie)
            gif_label.setFixedSize(500, 200)
            movie.start()
            layout.addWidget(gif_label, alignment=QtCore.Qt.AlignCenter)
        except:
            try:
                logo = QtGui.QPixmap(r":\IGIB_LOGO.png").scaled(200, 200, QtCore.Qt.KeepAspectRatio)
                logo_label = QtWidgets.QLabel()
                logo_label.setPixmap(logo)
                layout.addWidget(logo_label, alignment=QtCore.Qt.AlignCenter)
            except:
                pass

        title_label = QtWidgets.QLabel("Welcome to Drowsiness Detection Test")
        title_label.setProperty("role", "title")
        title_label.setFont(QtGui.QFont("Space Grotesk", 16, QtGui.QFont.Bold))
        layout.addWidget(title_label, alignment=QtCore.Qt.AlignCenter)
        layout.addSpacerItem(QtWidgets.QSpacerItem(20, 20))
        
        desc_label = QtWidgets.QLabel(
            "This app uses webcam to detect blinks and assess drowsiness with EAR and PERCLOS algorithms."
        )
        desc_label.setProperty("role", "panel")
        desc_label.setFont(QtGui.QFont("Space Grotesk", 10, QtGui.QFont.Bold))
        desc_label.setWordWrap(True)
        desc_label.setFixedWidth(500)
        layout.addWidget(desc_label, alignment=QtCore.Qt.AlignCenter)
        
        layout.addSpacerItem(QtWidgets.QSpacerItem(50, 50))
        next_button = QtWidgets.QPushButton("Next")
        next_button.setFont(QtGui.QFont("Space Grotesk", 12, QtGui.QFont.Bold))
        next_button.clicked.connect(self.show_user_type_page)
        layout.addWidget(next_button, alignment=QtCore.Qt.AlignCenter)

        self._update_page(widget)

    # ---------------- PAGE 2 ----------------
    def show_user_type_page(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setAlignment(QtCore.Qt.AlignCenter)

        try:
             movie = QtGui.QMovie(r":\gifs\2smooth_loop.gif") #second transition
             background = QtWidgets.QLabel(widget)
             background.setMovie(movie)
             background.setGeometry(0, 0, 800, 800)
             movie.start()
             background.lower()
        except:
            widget.setStyleSheet("background-color: #f0f0f0;")

        logo_label = QtWidgets.QLabel()
        logo_pixmap = QtGui.QPixmap(r":/images/csir.png").scaled(200, 200, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        logo_label.setPixmap(logo_pixmap)
        layout.addWidget(logo_label, alignment=QtCore.Qt.AlignCenter)
        layout.addSpacerItem(QtWidgets.QSpacerItem(20, 20))

        title = QtWidgets.QLabel("HELLO, HOW ARE YOU?😀")
        
        title.setProperty("role", "title")
        title.setFont(QtGui.QFont("Space Grotesk", 12, QtGui.QFont.Bold))
        layout.addWidget(title, alignment=QtCore.Qt.AlignCenter)
        layout.addSpacerItem(QtWidgets.QSpacerItem(30, 30))
        # New user
        new_panel = QtWidgets.QLabel("Are you a new user?")
        new_panel.setFont(QtGui.QFont("Space Grotesk", 12, QtGui.QFont.Bold))
        new_btn = QtWidgets.QPushButton("New User")
        new_btn.clicked.connect(self.handle_new_user_selected)
        layout.addWidget(new_panel, alignment=QtCore.Qt.AlignCenter)
        layout.addWidget(new_btn, alignment=QtCore.Qt.AlignCenter)
        layout.addSpacerItem(QtWidgets.QSpacerItem(20, 20))
        
        # Existing user
        exist_panel = QtWidgets.QLabel("Continue as an existing user")
        exist_panel.setFont(QtGui.QFont("Space Grotesk", 12, QtGui.QFont.Bold))
        exist_btn = QtWidgets.QPushButton("Existing User")
        exist_btn.clicked.connect(self.handle_existing_user_selected)
        layout.addWidget(exist_panel, alignment=QtCore.Qt.AlignCenter)
        layout.addWidget(exist_btn, alignment=QtCore.Qt.AlignCenter)
        nav_layout = QtWidgets.QHBoxLayout()
        nav_layout.addStretch(1)
        layout.addSpacerItem(QtWidgets.QSpacerItem(30, 30))

        prev_btn = QtWidgets.QPushButton("Previous")
        prev_btn.clicked.connect(self.show_welcome_page)
        nav_layout.addWidget(prev_btn)
        nav_layout.addStretch(1)
        layout.addLayout(nav_layout) 
        self._update_page(widget)
        
        layout.addLayout(nav_layout)

        self._update_page(widget)

    # Handlers for user type selection
    def handle_new_user_selected(self):
        self.user_selected = True
        self.user_type = "new"
        self.show_new_user_page()

    def handle_existing_user_selected(self):
        self.user_selected = True
        self.user_type = "existing"
        self.show_existing_user_page()
        
    # Handler for Next button on page 2
    

    # ---------------- PAGE 3 NEW USER ----------------
    def show_new_user_page(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setAlignment(QtCore.Qt.AlignCenter)

        try:
             movie = QtGui.QMovie(r":\gifs\3smooth_loop.gif") #third transition
             background = QtWidgets.QLabel(widget)
             background.setMovie(movie)
             background.setGeometry(0, 0, 800, 800)
             movie.start()
             background.lower()
        except:
            widget.setStyleSheet("background-color: #f0f0f0;")

        title = QtWidgets.QLabel("Register as New User")
        title.setProperty("role", "title")
        title.setFont(QtGui.QFont("Space Grotesk", 18, QtGui.QFont.Bold))
        layout.addWidget(title, alignment=QtCore.Qt.AlignCenter)
        layout.addSpacerItem(QtWidgets.QSpacerItem(30, 30))
        
       
    
        name_label = QtWidgets.QLabel("Enter your name:")
        name_label.setProperty("role", "panel")
        name_label.setFont(QtGui.QFont("Space Grotesk", 14, QtGui.QFont.Bold))
        layout.addWidget(name_label, alignment=QtCore.Qt.AlignCenter)
        layout.addSpacerItem(QtWidgets.QSpacerItem(10, 10))
        name_entry = QtWidgets.QLineEdit()
        name_entry.setFixedWidth(350)
        name_entry.setPlaceholderText("Your name (e.g., JOHN)")
        layout.addWidget(name_entry, alignment=QtCore.Qt.AlignCenter)
        
        id_label = QtWidgets.QLabel("Enter your Unique ID:")
        id_label.setProperty("role", "panel")
        id_label.setFont(QtGui.QFont("Space Grotesk", 14, QtGui.QFont.Bold))
        layout.addWidget(id_label, alignment=QtCore.Qt.AlignCenter)
        layout.addSpacerItem(QtWidgets.QSpacerItem(10, 10))
        
        id_entry = QtWidgets.QLineEdit()
        id_entry.setFixedWidth(350)
        layout.addWidget(id_entry, alignment=QtCore.Qt.AlignCenter)
        id_entry.setPlaceholderText("Enter Unique ID (Special & Uppercase Allowed)")
        id_entry.setEchoMode(QtWidgets.QLineEdit.Password)

        # Add eye icon toggle for password visibility
        eye_btn = QtWidgets.QPushButton()
        eye_btn.setCheckable(True)
        eye_btn.setFixedSize(50, 50)
        eye_btn.setIcon(QtGui.QIcon(r":/images/eyeyey.png"))
        eye_btn.setStyleSheet("border: none; background: transparent;")

        def toggle_eye():
            if eye_btn.isChecked():
                id_entry.setEchoMode(QtWidgets.QLineEdit.Normal)
            else:
                id_entry.setEchoMode(QtWidgets.QLineEdit.Password)
        eye_btn.toggled.connect(toggle_eye)

        id_layout = QtWidgets.QHBoxLayout()
        id_layout.addWidget(id_entry)
        id_layout.addWidget(eye_btn)

        layout.addWidget(name_label, alignment=QtCore.Qt.AlignLeft)
        layout.addWidget(name_entry)
        layout.addWidget(id_label, alignment=QtCore.Qt.AlignLeft)
        layout.addLayout(id_layout)

        def save_data():
            name = name_entry.text().strip()
            uid = id_entry.text().strip()
            if not name or not uid:
                QtWidgets.QMessageBox.warning(self, "Error", "Name and ID required!")
                return
            # NEW: Prevent duplicate IDs
            if id_exists(uid):
                QtWidgets.QMessageBox.warning(self, "Duplicate ID", "This ID already exists. Please choose a different ID.")
                return
            saved = append_new_user(name, uid)
            print(f"[DEBUG] Registration for {uid} success: {saved}")

            print("[DEBUG] Registry Users After Registration:", [r.get("ID") for r in read_all_users()])
            # result will be added later
            if not saved:
                QtWidgets.QMessageBox.warning(self, "Duplicate ID", "This ID already exists or could not be saved. Please choose a different ID.")
                return
            self.current_user_id = uid 
            self.current_user_name = name 
            self.show_test_selection_page()
            
        layout.addSpacerItem(QtWidgets.QSpacerItem(20, 20))
        nav_layout = QtWidgets.QHBoxLayout()
        prev_btn = QtWidgets.QPushButton("Previous")
        prev_btn.clicked.connect(self.show_user_type_page)
        layout.addSpacerItem(QtWidgets.QSpacerItem(20, 20))
        save_btn = QtWidgets.QPushButton("Save and Continue")
        save_btn.clicked.connect(save_data)
        nav_layout.addWidget(prev_btn)
        nav_layout.addWidget(save_btn)
    
        layout.addLayout(nav_layout)
        self._update_page(widget)

    # ---------------- PAGE 3 EXISTING USER ----------------
    def show_existing_user_page(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setAlignment(QtCore.Qt.AlignCenter)
        try:
             movie = QtGui.QMovie(r":\gifs\3smooth_loop.gif") #4th transition
             background = QtWidgets.QLabel(widget)
             background.setMovie(movie)
             background.setGeometry(0, 0, 800, 800)
             movie.start()
             background.lower()
        except:
            widget.setStyleSheet("background-color: #f0f0f0;")

        title = QtWidgets.QLabel("Existing User Login")
        title.setProperty("role", "title")
        title.setFont(QtGui.QFont("Space Grotesk", 18, QtGui.QFont.Bold))
        layout.addWidget(title, alignment=QtCore.Qt.AlignCenter)
        layout.addSpacerItem(QtWidgets.QSpacerItem(30, 30))
        
        self.id_entry = QtWidgets.QLineEdit()
        self.id_entry.setFixedWidth(350)
        self.id_entry.setPlaceholderText("Enter your Unique ID")
        self.id_entry.setFont(QtGui.QFont("Space Grotesk", 12, QtGui.QFont.Bold))
        self.id_entry.setEchoMode(QtWidgets.QLineEdit.Password)
        layout.addWidget(self.id_entry, alignment=QtCore.Qt.AlignCenter)

        eye_btn = QtWidgets.QPushButton()
        eye_btn.setCheckable(True)
        eye_btn.setFixedSize(50, 50)
        eye_btn.setIcon(QtGui.QIcon(r":/images/eyeyey.png"))  # Use your eye icon path
        eye_btn.setStyleSheet("border: none; background: transparent;")
           
        def toggle_eye():
            if eye_btn.isChecked():
                self.id_entry.setEchoMode(QtWidgets.QLineEdit.Normal)
            else:
                self.id_entry.setEchoMode(QtWidgets.QLineEdit.Password)
        eye_btn.toggled.connect(toggle_eye)

        id_layout = QtWidgets.QHBoxLayout()
        id_layout.addWidget(self.id_entry)
        id_layout.addWidget(eye_btn)
        layout.addLayout(id_layout)
        
        def verify_user():
            uid = self.id_entry.text().strip()
            if not uid:
                QtWidgets.QMessageBox.warning(self, "Error", "ID required!")
                return
            # NEW: Verify against CSV
            if not id_exists(uid):
                QtWidgets.QMessageBox.warning(self, "User Not Found", "No user with this ID exists. Please register as a new user.")
                return  # block proceed
            self.current_user_id = uid
            self.current_user_name = ""
            users = read_all_users()
            for u in users:
                if u["ID"] == uid:
                    self.current_user_name = u.get("name", "")
                    break
            self.show_test_selection_page()
            

        nav_layout = QtWidgets.QHBoxLayout()
        prev_btn = QtWidgets.QPushButton("Previous")
        prev_btn.clicked.connect(self.show_user_type_page)
        layout.addSpacerItem(QtWidgets.QSpacerItem(30, 30))
        next_btn = QtWidgets.QPushButton("Continue")
        next_btn.clicked.connect(verify_user)
        nav_layout.addWidget(prev_btn)
        nav_layout.addWidget(next_btn)

        layout.addLayout(nav_layout)
        self._update_page(widget)
    
    # ---------------- PAGE 4: TEST SELECTION ----------------
    def show_test_selection_page(self):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setAlignment(QtCore.Qt.AlignCenter)
        try:
             movie = QtGui.QMovie(r":\gifs\6smooth_loop.gif")
             background = QtWidgets.QLabel(widget)
             background.setMovie(movie)
             background.setGeometry(0, 0, 800, 800)
             movie.start()
             background.lower()
        except:
            widget.setStyleSheet("background-color: #f0f0f0;")
            
            
        title = QtWidgets.QLabel("Please Choose a Test")
        title.setProperty("role", "title")
        title.setFont(QtGui.QFont("Space Grotesk", 18, QtGui.QFont.Bold))
        title.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(title)
        
        n_panel = (
            "<ul>"
            "<li>✅<b>Baseline Test:</b> Register/update your personal baseline (stored)</li>"
            "<li>🔄<b>Random Test:</b> One-time test using standard threshold (16 blinks/min)</li>"
            "<li>🔃<b>After Baseline Test:</b> Test using YOUR stored baseline for personalized results</li>"
            "</ul>"
        )
        n_panel = QtWidgets.QLabel(n_panel)
        n_panel.setProperty("role", "panel")
        n_panel.setFont(QtGui.QFont("Space Grotesk", 9, QtGui.QFont.Bold))
        n_panel.setTextFormat(QtCore.Qt.RichText)
        layout.addWidget(n_panel)
        layout.addSpacerItem(QtWidgets.QSpacerItem(80, 80))

        baseline_btn = QtWidgets.QPushButton("Register your baseline")
        baseline_btn.clicked.connect(lambda: self.open_test_window("baseline"))
        layout.addWidget(baseline_btn, alignment=QtCore.Qt.AlignCenter)
        
        layout.addSpacerItem(QtWidgets.QSpacerItem(80, 80))
        
        testing_btn = QtWidgets.QPushButton("Testing (Random)")
        testing_btn.clicked.connect(lambda: self.open_test_window("testing"))
        layout.addWidget(testing_btn, alignment=QtCore.Qt.AlignCenter)
        
        layout.addSpacerItem(QtWidgets.QSpacerItem(80, 80))
        
        after_baseline_btn = QtWidgets.QPushButton("🔍 After Baseline Test")
        after_baseline_btn.clicked.connect(lambda: self.open_test_window("after_baseline"))
        layout.addWidget(after_baseline_btn, alignment=QtCore.Qt.AlignCenter)

        layout.addSpacerItem(QtWidgets.QSpacerItem(40, 40))
        nav_layout = QtWidgets.QHBoxLayout()
        prev_btn = QtWidgets.QPushButton("Previous")
        layout.addSpacerItem(QtWidgets.QSpacerItem(30, 30))
        prev_btn.clicked.connect(self.show_user_type_page)
        next_btn = QtWidgets.QPushButton("Next")
        next_btn.clicked.connect(lambda: QtWidgets.QMessageBox.information(self, "Next", "Please click on existing options"))
        nav_layout.addWidget(prev_btn)
        nav_layout.addWidget(next_btn)

        layout.addLayout(nav_layout)
        self._update_page(widget)
    
    
    def get_user_baseline_average(self,user_id):
        rows = read_all_users()
        baseline_values =[]
        for row in rows:
            if (row.get("ID") or "").strip() == user_id:
                baseline_str = row.get("personal_baseline_bpm", "") or row.get("baseline_per_min", "")
                if baseline_str:
                    try:
                        baselines = [float(x.strip()) for x in baseline_str.split(",") if x.strip()]
                        baseline_values.extend(baselines)
                    except Exception:
                        try:
                            baseline_values.append(float(baseline_str))
                        except:
                            pass

        if not baseline_values:
             print(f"[WARN] No baseline values found for user {user_id}")
             return None
        recent_baselines = baseline_values[-5:] if len(baseline_values) >= 5 else baseline_values
        if not recent_baselines:
           print(f"[WARN] Recent baselines empty for user {user_id}")
           return None
        avg_baseline = sum(recent_baselines) / len(recent_baselines)
        print(f"[INFO] User {user_id} baseline average (last {len(recent_baselines)}): {avg_baseline:.2f}")
        return avg_baseline
       
    def open_test_window(self, mode):
        if mode == "testing":
            mode = "random"
        self.mode = mode
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setAlignment(QtCore.Qt.AlignCenter)
        try:
             movie = QtGui.QMovie(r":\gifs\6smooth_loop.gif") #third transition
             background = QtWidgets.QLabel(widget)
             background.setMovie(movie)
             background.setGeometry(0, 0, 800, 800)
             movie.start()
             background.lower()
        except:
            widget.setStyleSheet("background-color: #f0f0f0;")
        
    
        layout.addSpacerItem(QtWidgets.QSpacerItem(10, 10))
 
        title = QtWidgets.QLabel("Test Window (2.15 minutes)")
        title.setProperty("role", "title")
        title.setFont(QtGui.QFont("Space Grotesk", 20, QtGui.QFont.Bold))
        layout.addWidget(title, alignment=QtCore.Qt.AlignCenter)
        layout.addSpacerItem(QtWidgets.QSpacerItem(20, 20))
        try:
            movie_gif = QtGui.QMovie(r":/gifs/eyeyeeymp4.gif")
            gif_label = QtWidgets.QLabel(widget)
            gif_label.setMovie(movie_gif)
            gif_label.setScaledContents(True)  

            movie_gif.setScaledSize(QtCore.QSize(400, 300))
            gif_label.setMovie(movie_gif)
            movie_gif.start()

            layout.addWidget(gif_label, alignment=QtCore.Qt.AlignCenter)
            
            gif_container = QtWidgets.QWidget()
            gif_layout = QtWidgets.QVBoxLayout(gif_container)
            gif_layout.addWidget(gif_label, alignment=QtCore.Qt.AlignCenter)
            layout.addWidget(gif_container, alignment=QtCore.Qt.AlignCenter)
            
        except Exception as e:
            print("GIF Error:", e)
        
        layout.addSpacerItem(QtWidgets.QSpacerItem(30, 30))
        
        sub_text = """
         <ul>
            <li>✅Click start to start the test</li>
            <li>👨🏻Sit upright and keep the face in frame</li>
            <li>😀Stay neutral as possible</li>
        </ul>
        """
        sub = QtWidgets.QLabel(sub_text)
        sub.setProperty("role", "panel")
        sub_font = QtGui.QFont()
        sub_font.setBold(True)
        sub_font.setPointSize(11)
        sub.setFont(sub_font)
        sub.setTextFormat(QtCore.Qt.RichText)
        sub.setAlignment(QtCore.Qt.AlignCenter)
       # Enables HTML formatting
        sub.setWordWrap(True)
        layout.addWidget(sub)

        layout.addSpacerItem(QtWidgets.QSpacerItem(20, 20))
        start_btn = QtWidgets.QPushButton("START")
        start_btn.setMinimumHeight(120)
        start_btn.setMinimumWidth(300)
        start_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF0000;
                color: white;
                font-size: 36px;
                font-weight: bold;
                border-radius: 12px;
                padding: 20px 40px;
            }
            QPushButton:hover { background-color: #28a745; }
        """)
        layout.addWidget(start_btn, alignment=QtCore.Qt.AlignCenter)
        
        layout.addSpacerItem(QtWidgets.QSpacerItem(50, 50))

        nav_layout = QtWidgets.QHBoxLayout()
        prev_btn = QtWidgets.QPushButton("Previous")
        prev_btn.clicked.connect(self.show_test_selection_page)
        nav_layout.addWidget(prev_btn)
        layout.addLayout(nav_layout)

        def start_test():
            start_btn.setDisabled(True)
            self.run_detection()

        start_btn.clicked.connect(start_test)

        self._update_page(widget)

    # ============== THREAD HANDLERS ==============
    def run_detection(self):
        user_baseline = None
        
        if self.mode == "after_baseline":
           user_id = self.current_user_id
           if not user_id:
               QtWidgets.QMessageBox.warning(self, "Warning", 
                "User ID not found. Cannot run After Baseline test.")
               return
           
           user_baseline = self.get_user_baseline_average(user_id)
           if user_baseline is None or user_baseline <= 0:
               QtWidgets.QMessageBox.warning(self, "No Baseline Found", 
                f"User {user_id} has no baseline data.\nPlease run a Baseline Test first.")
               return
        
        # Prepare worker and thread
        self.thread = QtCore.QThread()
        self.worker = BlinkAccessor(mode=self.mode, user_baseline=user_baseline)
        self.worker.moveToThread(self.thread)

        # Connect signals
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_test_finished)
        self.worker.error.connect(self.on_test_error)
        self.worker.finished.connect(self.thread.quit)
        self.worker.error.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def on_test_error(self, msg):
        QtWidgets.QMessageBox.critical(self, "Error", msg)
    ############################################################
    def on_test_finished(self, result, test_start_time=None):
        try:
            user_id = getattr(self, "current_user_id", None)
            if not user_id:
               QtWidgets.QMessageBox.warning(self, "Warning", "User ID not found. Test results will not be saved.")
               return
            if self.mode == "random":
                for checkpoint in [145, 115, 85]:
                    cpr = result.get("checkpoint_results", {}).get(checkpoint, {})
                    blink_rate_class = cpr.get("classification_blink_rate", {})
                    if "status" not in blink_rate_class:
                        blink_rate_class["status"] = "NotAssessed"  # Or your desired default
                        cpr["classification_blink_rate"] = blink_rate_class
                        result["checkpoint_results"][checkpoint] = cpr
            baselines = self.get_baseline_list(user_id)
            user_dict = {
            "name": getattr(self, "current_user_name", "UnknownUser"),
            "id": user_id
            }
            test_start_time = test_start_time or getattr(self, "test_start_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            print(f"[DEBUG] Saving test result to CSV for user {user_dict['id']} ({self.mode})")
            save_test_result(user_dict, result, self.mode, test_start_time)
            if self.mode == "baseline":
               baseline_bpm = result.get("personal_baseline_bpm", None)
               if baseline_bpm is not None:
                  append_baseline_to_registry(user_id, baseline_bpm)
            status = "Unknown"
            checkpoint_results = result.get("checkpoint_results", {})
            for checkpoint_time in [145, 115, 85]:
                if checkpoint_time in checkpoint_results:
                   checkpoint_data = checkpoint_results[checkpoint_time]
                   blink_rate_class = checkpoint_data.get("classification_blink_rate", {})
                   status = blink_rate_class.get("status", "Unknown")
                   if status != "Unknown":
                        print(f"[INFO] Status extracted from checkpoint {checkpoint_time}s: {status}")
                        break
            if status in ["Unknown", "NotAssessed", "", None]:
                print("[DEBUG] No valid status found for random test.")
                status = "NotAssessed"
                print("[DEBUG] Defaulting to NotAssessed.")
                
            summary = [
            f"Mode: {self.mode}",
            f"Total Blinks: {result.get('total_blinks', 0)}"
            ]
            baseline_val = result.get("baseline_per_min", 0.0)
            if baseline_val > 0:
               summary.append(f"Personal Baseline: {baseline_val:.2f}/min")
            ear_threshold = result.get("ear_threshold", None)
            if ear_threshold:
               summary.append(f"EAR Threshold: {ear_threshold:.3f}")
            QtWidgets.QMessageBox.information(self, "Test Results", "\n".join(summary))
            self.show_conclusion_page(status=status)
        except Exception as e:
           print(f"[ERROR] Exception in on_test_finished: {e}")
           QtWidgets.QMessageBox.critical(self, "Error", f"Could not save results: {str(e)}")

    # ---------------- PAGE 6: CONCLUSION ----------------
       
    def show_conclusion_page(self,status=""):
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setAlignment(QtCore.Qt.AlignCenter)
        try:
             movie = QtGui.QMovie(r":\gifs\7smooth_loop.gif") 
             background = QtWidgets.QLabel(widget)
             background.setMovie(movie)
             background.setGeometry(0, 0, 800, 800)
             movie.start()
             background.lower()
        except:
            widget.setStyleSheet("background-color: #f0f0f0;")

        # Logo
        logo_label = QtWidgets.QLabel()
        logo_pixmap = QtGui.QPixmap(r":/images/csir.png").scaled(200, 200, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        logo_label.setPixmap(logo_pixmap)
        layout.addWidget(logo_label, alignment=QtCore.Qt.AlignCenter)
        layout.addSpacerItem(QtWidgets.QSpacerItem(50, 50))

        # Thank you message
        thank_label = QtWidgets.QLabel("Thank you for your time!")
        thank_label.setFont(QtGui.QFont("Space Grotesk", 18, QtGui.QFont.Bold))
        thank_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(thank_label, alignment=QtCore.Qt.AlignCenter)
        layout.addSpacerItem(QtWidgets.QSpacerItem(50, 50))
        
        status_label = QtWidgets.QLabel()
        status_label.setFont(QtGui.QFont("Space Grotesk", 11, QtGui.QFont.Bold))
        status_label.setAlignment(QtCore.Qt.AlignCenter)
        
        msg = ""
        if status.lower() == "drowsy":
           msg = "Warning: you might be drowsy, if tested baseline- are you sure you want register this as baseline?"
        elif status.lower() == "alert":
           msg = "Status: Alert - you might be alert."
        elif status.lower() == "dicey":
           msg = "Status: Borderline - you may be getting drowsy."
        elif status.lower() == "neutral":
           msg = "Status: Neutral - keep monitoring yourself."
        else:
           msg = f"Status: {status}"
           
        status_label.setText(msg)
        layout.addWidget(status_label, alignment=QtCore.Qt.AlignCenter)
        layout.addSpacerItem(QtWidgets.QSpacerItem(20, 20))
        # Start Over button - green
        start_over_btn = QtWidgets.QPushButton("Start Over")
        start_over_btn.setStyleSheet("""
        QPushButton {
            background-color: #28a745;  /* green */
            color: white;
            padding: 12px 40px;
            border: none;
            border-radius: 5px;
            font-weight: bold;
            font-size: 16px;
        }
        QPushButton:hover {
            background-color: #218838;
        }
    """)
        start_over_btn.clicked.connect(self.show_welcome_page)
        layout.addWidget(start_over_btn, alignment=QtCore.Qt.AlignCenter)

        self._update_page(widget)

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = WelcomeApp()
    window.show()
    sys.exit(app.exec_())
