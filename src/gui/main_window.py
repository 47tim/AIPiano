from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QRadioButton, QComboBox, QLabel, QGroupBox, QSpinBox, QTabWidget, QFileDialog, 
)
from PySide6.QtGui import QPainter, QColor, QBrush, QPalette
import sys
import subprocess
import time
from pathlib import Path
import threading
import pygame
import pretty_midi
from PySide6.QtCore import Qt, QTimer, QSize, Signal, QObject, QThread
from PySide6.QtWidgets import QFormLayout, QProgressBar, QStyle, QToolButton
import re

# Timothy Hyde 2025
# This file is the main file of the GUI, using PySide6, PyGame for playing the MIDI files, and PrettyMIDI for visualizing the MIDI files. 

# Class for visualizing the MIDI output in a piano-roll type of setup
class MidiVisualizer(QWidget):
    
    def __init__(self):
        super().__init__()
        self.midi = None
        self.max_time = 1.0          
        self.playhead = 0.0         
        
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, False)
        self.setAttribute(Qt.WidgetAttribute.WA_PaintOnScreen, False)
        self.setAutoFillBackground(False)

   
    def set_playhead(self, seconds: float):
        self.playhead = seconds
        self.update()              

    def load_midi(self, midi_path):
        try:
            self.midi = pretty_midi.PrettyMIDI(midi_path)
    
            self.max_time = max((n.end for inst in self.midi.instruments
                                 for n in inst.notes), default=1.0)
            self.playhead = 0.0
            QTimer.singleShot(0, self.update)
        except Exception as e:
            print(f"Failed to load MIDI: {e}")

    def paintEvent(self, event):
        if not self.midi:
            return
        painter = QPainter(self)
        # Turned on antialiasing to remove the stutter from the playhead line
        painter.setRenderHint(QPainter.Antialiasing) 
        painter.fillRect(self.rect(), QColor(30, 30, 30))

        w, h = self.width(), self.height()
        max_t = self.max_time or 1.0           




        for inst in self.midi.instruments:
            if inst.is_drum:
                continue
            for n in inst.notes:
                if n.start > max_t:
                    continue
                x1 = (n.start / max_t) * w
                x2 = (n.end   / max_t) * w
                y  = h - ((n.pitch - 21) / 87) * h
                painter.setBrush(QColor(50 + 205*n.velocity/127,
                                        100 + 100*(1 - n.velocity/127),
                                        250))
                painter.setPen(Qt.NoPen)
                painter.drawRect(int(x1), int(y)-4, max(2, int(x2-x1)), 8)

     
        head_x = int((self.playhead / max_t) * w)
        painter.setPen(QColor(100, 149, 237))
        painter.drawLine(head_x, 0, head_x, h)


# This class handles the actual generation, chooses which python script to call depending on which of the 6 models are chosen
class Generator(QObject):
    
    progress = Signal(int)
    finished = Signal(str)
    status = Signal(str)

    def __init__(self, args):
        super().__init__()
        self.args = args

    def run(self):
        model_type, model_size, prompt, top_k, top_p, save_path = self.args
        if not prompt.strip():
            self.status.emit("Please enter a prompt")
            return

        checkpoint_folder = Path(__file__).resolve().parents[2] / "checkpoints" / model_type / model_size
        all_ckpts = list(checkpoint_folder.glob("music_transformer_epoch*.pt"))
        if not all_ckpts:
            self.status.emit(f"No checkpoints found at: {checkpoint_folder}")
            return

        latest = max(all_ckpts, key=lambda p: int(p.stem.split("_epoch")[-1]))
        output = save_path or f"outputs/{model_type}_{model_size}_{int(time.time())}.mid"

        # Prompt format, same as in generate.py.
        cmd = [
            sys.executable, "-m", f"src.{model_type}.generate",
            "--prompt", prompt,
            "--checkpoint", str(latest),
            "--output", output,
            "--model_size", model_size,
            "--top_k", str(top_k),
            "--top_p", str(top_p)
        ]

        # Was having problems with STD reading in non standard chars from TQDM's command line in my generation scripts
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1
        )
        for line in self.proc.stdout:
            if m := re.search(r"(\d+)%\|", line):
                self.progress.emit(int(m.group(1)))
        self.proc.wait()
        if Path(output).exists():
            self.finished.emit(output)
        else:
            self.status.emit("Generation cancelled")
            
    def cancel(self):
        if hasattr(self, 'proc') and self.proc and self.proc.poll() is None:
            self.proc.terminate()




class MainWindow(QMainWindow):
    
    def __init__(self):
        
        super().__init__()
        
        # NEW -- might need to remove if giving issues on other operating systems
        self.setWindowFlag(Qt.FramelessWindowHint)


        self.generation_thread = None
        self.generation_process = None
        self.is_generating = False


        self.setWindowTitle("PianoMuse")
        self.setFixedSize(800, 900)

        # Setting main tab
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        

        # Two pages, might add third for info
        self.generation_page = self.build_generation_page()
        self.playback_page = self.build_playback_page()

        self.tabs.addTab(self.generation_page, "Generate")
        self.tabs.addTab(self.playback_page, "Playback")
        
        # NEW
        # putting in a fake spacer tab in between my two tabs, and my fake x and - buttons. Unfortunately with this option, users cannot 
        # drag the window from the top. Also, screens with different DPI's will have issues with the amount of space I hardcoded. 
        # Definitley need to make this more suitable for all screen sizes and DPIs.
        fake_spacer = QWidget()
        self.tabs.addTab(fake_spacer, "                                                                                        ")
        self.tabs.tabBar().setTabEnabled(2, False)  
        

        fake_minimize = QWidget()
        fake_close = QWidget()
        self.tabs.addTab(fake_minimize, "-")
        self.tabs.addTab(fake_close, "×")


        self.tabs.currentChanged.connect(self.handle_tab_change)
        
        
     
      
    # UI for generation page
    def build_generation_page(self):
        
        page = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(15) 
        layout.setContentsMargins(60, 30, 60, 30)  
        
        title = QLabel("PianoMuse")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 50px; font-weight: bold; margin-bottom: 20px; color: cornflowerblue")
        layout.addWidget(title)

       
        # model select
        model_layout = QVBoxLayout()

        model_label = QLabel("Model")
        model_label.setStyleSheet("font-size: 18px; color: #7c7c80")
        model_layout.addWidget(model_label)

        radio_layout = QHBoxLayout()
        self.performance_radio = QRadioButton("Performance-Based")
        self.notation_radio = QRadioButton("Notation-Based")
        self.performance_radio.setChecked(True)
        radio_layout.addWidget(self.performance_radio)
        radio_layout.addWidget(self.notation_radio)

        model_layout.addLayout(radio_layout)
        layout.addLayout(model_layout)

        #7c7c80
        

        # model size
        size_layout = QVBoxLayout()

        size_label = QLabel("Model Size")
        size_label.setStyleSheet("color: #7c7c80")
        size_layout.addWidget(size_label)

        self.size_dropdown = QComboBox()
        self.size_dropdown.addItems(["Low", "Medium", "High"])
        self.size_dropdown.setStyleSheet("""
            QComboBox {
                border: 0px solid #181a1c;
                border-radius: 6px;
                padding: 5px;
                color: white;
            }
            QComboBox::drop-down {
                border: 0px solid #181a1c;
            }
        """)


        size_layout.addWidget(self.size_dropdown)

        layout.addLayout(size_layout)

        # prompt input label
        prompt_layout = QVBoxLayout()
        prompt_label = QLabel("Prompt")
        prompt_label.setStyleSheet("font-size: 18px; color: #7c7c80")

        prompt_layout.addWidget(prompt_label)
        self.prompt_input = QLineEdit()
        self.prompt_input.setPlaceholderText("Enter your music prompt here...")
        self.prompt_input.setStyleSheet("""
            QLineEdit {
                border: 0px solid #181a1c;
                border-radius: 6px;
                padding: 5px;
                color: white;
            }
        """)
        prompt_layout.addWidget(self.prompt_input)
        layout.addLayout(prompt_layout)
        self.prompt_input.setToolTip(
            "For best results, be descriptive and try to reference speed, emotion, and even composer style \n"
            "Ex: 'Slow, dreamy, dark and melancholic waltz by Chopin with elegant chords and structure'"
        )
        

        # generate button, turns to cancel during generation
        self.generate_button = QPushButton("Generate")
        self.generate_button.setMinimumHeight(40)
        self.generate_button.clicked.connect(self.start_generation)
        layout.addWidget(self.generate_button)
        self.generate_button.setStyleSheet("""
            QPushButton {
                background-color: cornflowerblue;
                color: white;
                font-weight: bold;
                font-size: 18px;
                border: none;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #218838; 
            }
        """)

        # advanced settings dropdown (contains top p and top k settings)
        self.advanced_button = QPushButton("Show Advanced Settings")
        self.advanced_button.setCheckable(True)
        self.advanced_button.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                padding: 6px;
                border: 0px solid #181a1c;
                border-radius: 6px;
                background-color: #252629;
                color: white;
            }
            QPushButton:hover {
                background-color: #404040;
            }
        """)
        self.advanced_button.clicked.connect(lambda: self.toggle_advanced(additional_group))
        layout.addWidget(self.advanced_button)

        additional_group = QGroupBox("")
        additional_group.setStyleSheet("""
            QGroupBox {
                font-size: 12px; 
                color: #373737; 
                border: none;   
            }
        """)
        additional_layout = QHBoxLayout()  
        
        # top k
        top_k_label = QLabel("Top-K (%)")
        top_k_label.setStyleSheet("color: #7c7c80")
        top_k_layout = QVBoxLayout()
        top_k_layout.addWidget(top_k_label)
        
        self.top_k_spinbox = QSpinBox()
        self.top_k_spinbox.setRange(1, 1000)
        self.top_k_spinbox.setValue(50)
        self.top_k_spinbox.setStyleSheet ("""
            border: 0px solid #181a1c;
            border-radius: 6px;
            padding: 5px;
        
        """)
        top_k_layout.addWidget(self.top_k_spinbox)
        top_k_label.setToolTip("Higher value = more randomness. Lower value = more focused output.")

        # Top-P Section
        top_p_label = QLabel("Top-P (%)")
        top_p_label.setStyleSheet("color: #7c7c80")
        top_p_layout = QVBoxLayout()
        top_p_layout.addWidget(top_p_label)

        self.top_p_spinbox = QSpinBox()
        self.top_p_spinbox.setRange(1, 100)
        self.top_p_spinbox.setValue(95)
        self.top_p_spinbox.setStyleSheet ("""
            border: 0px solid #181a1c;
            border-radius: 6px;
            padding: 5px;
        
        """)
        top_p_layout.addWidget(self.top_p_spinbox)
        top_p_label.setToolTip("Higher value = more randomness. Lower value = safer but possibly dull output.")

       
        additional_layout.addLayout(top_k_layout)
        additional_layout.addSpacing(20) 
        additional_layout.addLayout(top_p_layout)

        additional_group.setLayout(additional_layout)
        layout.addWidget(additional_group)
        
        additional_group.setVisible(False)



        # saving destination
        save_layout = QHBoxLayout()

        self.save_path_input = QLineEdit()
        self.save_path_input.setPlaceholderText("Choose where to save the output MIDI...")
        self.save_path_input.setReadOnly(True)  
        self.save_path_input.setStyleSheet("""
            QLineEdit {
                border: 0px solid #181a1c;
                border-radius: 6px;
                background-color: #252629;
            }
            
        """)

        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self.browse_save_location)
        self.browse_button.setStyleSheet("""
            QPushButton {
                border: 0px solid #181a1c;
                border-radius: 6px;
                background-color: #252629;
            }
            QPushButton:hover {
                background-color: #505050;
            }
        """)

        save_layout.addWidget(self.save_path_input)
        save_layout.addWidget(self.browse_button)
        
        # border: 1px solid #181a1c; border-radius: 1px; background-color: #252629
        
        layout.addLayout(save_layout)


        # status bar
        self.status_label = QLabel("Ready to generate")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 24px; color: white; margin-top: 10px;")
        layout.addWidget(self.status_label)
        
        
        # progress bar which pulls from TQDM console output
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #181a1c;
                border-radius: 6px;
                background-color: #252629;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: cornflowerblue;
                width: 10px;
            }
        """)
        layout.addWidget(self.progress_bar)



        layout.addStretch(1)  

        page.setLayout(layout)
        
        return page

    
    # UI for MIDI playback page
    def build_playback_page(self):
        page = QWidget()
        layout = QVBoxLayout()

        self.midi_label = QLabel("No MIDI loaded")
        layout.addWidget(self.midi_label)

        self.visualizer = MidiVisualizer()
        self.visualizer.setMinimumHeight(600)
        layout.addWidget(self.visualizer)
        self.play_timer = QTimer(self)
        self.play_timer.setInterval(30)   # time in milliseconds               
        self.play_timer.timeout.connect(self._update_head)


        self.play_button = QPushButton("Play MIDI")
        self.play_button.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                padding: 6px;
                border: 0px solid #181a1c;
                border-radius: 6px;
                background-color: #252629;
                color: white;
            }
            QPushButton:hover {
                background-color: #404040;
            }
        """)
        self.stop_button = QPushButton("Stop MIDI")
        self.stop_button.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                padding: 6px;
                border: 0px solid #181a1c;
                border-radius: 6px;
                background-color: #252629;
                color: white;
            }
            QPushButton:hover {
                background-color: #404040;
            }
        """)
        self.load_button = QPushButton("Load MIDI File")
        self.load_button.setStyleSheet("""
            QPushButton {
                font-size: 16px;
                padding: 6px;
                border: 0px solid #181a1c;
                border-radius: 6px;
                background-color: #252629;
                color: white;
            }
            QPushButton:hover {
                background-color: #404040;
            }
        """)

        self.play_button.clicked.connect(self.play_midi)
        self.stop_button.clicked.connect(self.stop_midi)
        self.load_button.clicked.connect(self.load_midi)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.play_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.load_button)

        layout.addLayout(button_layout)
        page.setLayout(layout)
        return page




    # ----------------------------------------------------------------------------------------------------------------------------------------------- 
    # GENERATION LOGIC
    
    # handles the logic for starting generating in a seperate thread, so the UI doesn't freeze
    def start_generation(self):
        
        if self.is_generating:
            self.worker.cancel()                  
            self.worker_thread.quit()              
            self.worker_thread.wait()
            self.reset_ui_after_cancel()
            return

        self.generate_button.setText("Cancel")
        self.generate_button.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                font-weight: bold;
                font-size: 18px;
                border-radius: 8px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
        """)
        self.is_generating = True
        self.status_label.setText("Generating...")

        args = (
            'performance' if self.performance_radio.isChecked() else 'notation',
            self.size_dropdown.currentText().lower(),
            self.prompt_input.text(),
            self.top_k_spinbox.value(),
            self.top_p_spinbox.value() / 100,
            self.save_path_input.text()
        )

        self.worker       = Generator(args)
        self.worker_thread = QThread()
        self.worker.moveToThread(self.worker_thread)

        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.status.connect(self.status_label.setText)
        self.worker.finished.connect(self.on_finished)

        self.worker_thread.started.connect(self.worker.run)
        self.worker_thread.start()
        
    # resets the generate button back to default after generation is canceled
    def reset_ui_after_cancel(self):
        self.is_generating = False
        self.generate_button.setText("Generate")
        self.generate_button.setStyleSheet("""
            QPushButton {
                background-color: cornflowerblue;
                color: white;
                font-weight: bold;
                font-size: 18px;
                border: none;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        self.status_label.setText("Generation cancelled.")
        self.progress_bar.setValue(0)


    # handles the saving of the midi file and resets UI when complete
    def on_finished(self, filename):
        
        self.worker_thread.quit()
        self.worker_thread.wait()
        self.status_label.setText(f"Saved to: {filename}")
        self.progress_bar.setValue(0)
        
        self.reset_ui_after_cancel()

        self.worker_thread.quit()
        self.worker_thread.wait()

        self.status_label.setText(f"Saved to: {filename}")
        self.progress_bar.setValue(0)
        
        self.last_generated_midi = filename
        self.midi_label.setText(Path(filename).name)
        self.visualizer.load_midi(filename)
        
    # ----------------------------------------------------------------------------------------------------------------------------------------------- 
    
    def browse_save_location(self):
        file_dialog = QFileDialog(self)
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        file_dialog.setNameFilter("MIDI files (*.mid *.midi)")
        file_dialog.setDefaultSuffix("mid")
        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                self.save_path_input.setText(selected_files[0])
                
   
    def handle_tab_change(self, index):
        tab_text = self.tabs.tabText(index)
    
        if tab_text == "-":
            self.showMinimized()
            self.tabs.setCurrentIndex(0)  
    
        elif tab_text == "×":
            self.close()
            
    # ----------------------------------------------------------------------------------------------------------------------------------------------- 
    # USER INPUTS
    
    # I had to add these functions in, as with borderless windows there is no way to close, minimize, or drag the window around        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self.drag_position)
            event.accept()
    
    def toggle_advanced(self, group):
        if group.isVisible():
            group.setVisible(False)
            self.advanced_button.setText("Show Advanced Settings")
        else:
            group.setVisible(True)
            self.advanced_button.setText("Hide Advanced Settings")
            
    # ----------------------------------------------------------------------------------------------------------------------------------------------- 
    # PLAYBACK FUNCTIONS  

    # Moves the playhead along with the current position of the MIDI track playing
    def _update_head(self):
        if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
            pos_ms = pygame.mixer.music.get_pos()
            if pos_ms != -1:
                self.visualizer.set_playhead(pos_ms / 1000.0)
        else:             
            self.play_timer.stop()
            self.visualizer.set_playhead(0.0)

    # handles loading a midi from user's files
    def load_midi(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("MIDI files (*.mid *.midi)")
        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                self.last_generated_midi = selected_files[0]
                midi_name = Path(self.last_generated_midi).name
                self.midi_label.setText(f"{midi_name}")
                print(f"Loaded MIDI: {self.last_generated_midi}")
                self.visualizer.load_midi(self.last_generated_midi)


    # plays the midi using pygame's built in midi player functionality
    def play_midi(self):
        if not getattr(self, "last_generated_midi", None):
            print("No MIDI loaded.")
            return
        pygame.mixer.init()
        pygame.mixer.music.load(self.last_generated_midi)
        pygame.mixer.music.play()
        self.play_timer.start()         
        print(f"Playing MIDI: {self.last_generated_midi}")

    def stop_midi(self):
        pygame.mixer.music.stop()
        self.play_timer.stop()       
        self.visualizer.set_playhead(0.0)
        print("Stopped MIDI")


    # -----------------------------------------------------------------------------------------------------------------------------------------------   
    
    def reset_generate_button(self):
        self.generate_button.setText("Generate")
        self.generate_button.setStyleSheet("""
            QPushButton {
                background-color: cornflowerblue;
                color: white;
                font-weight: bold;
                font-size: 18px;
                border: none;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        self.is_generating = False



# -----------------------------------------------------------------------------------------------------------------------------------------------   


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Styling
    app.setStyleSheet("""
        QWidget {
            background-color: #1F2024;
            color: #dddddd;
            font-size: 18px;
        }
    
        QPushButton {
            background-color: #404040;
            border: 1px solid #555;
            padding: 8px;
            border-radius: 8px;
        }
        QPushButton:hover {
            background-color: #505050;
        }

        QLabel {
            border-radius: 6px; 
            padding: 2px;
        }

        QLineEdit, QComboBox, QSpinBox {
            background-color: #252629;
            border: 1px solid #555;
            padding: 4px;
            border-radius: 6px; 
        }

        QTabWidget::pane {
            border: 1px solid #333;
            padding: 5px;
        }

        QGroupBox {
            border: 1px solid #555;
            margin-top: 10px;
            border-radius: 8px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top center;
            padding: 0 3px;
        }

        QTabBar::tab {
            background: none; 
            border: none;     
            color: #cccccc;   
            padding: 10px 20px;
            font-size: 18px;
        }

        QTabBar::tab:selected {
            color: cornflowerblue; 
            border-bottom: 2px solid cornflowerblue; 
            margin-bottom: -2px;
        }

        QTabWidget::pane {
            border: none; 
            
        }
    """)


    window = MainWindow()
    window.show()
    sys.exit(app.exec())
