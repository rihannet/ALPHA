'''

ALPHA MAIN CODE 


'''
import sys
import pandas as pd
import joblib
import numpy as np
import os
import cv2
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from tensorflow.keras.models import load_model
from PyQt5 import QtCore, QtGui, QtWidgets, uic
import os
import cv2
import requests
from PyQt5 import QtWidgets, uic
import geocoder  # for getting the user's current location
import numpy as np
import torch
import torch.nn as nn
from PyQt5 import QtWidgets, uic



class LoginApp(QtWidgets.QMainWindow):
    def __init__(self):
        super(LoginApp, self).__init__()
        uic.loadUi('Login.ui', self)

        # Connect login and sign-up buttons
        self.loginButton = self.findChild(QtWidgets.QPushButton, 'BLOGIN')
        self.signupButton = self.findChild(QtWidgets.QPushButton, 'BSIGNUP')

        self.loginButton.clicked.connect(self.login)
        self.signupButton.clicked.connect(self.open_signup_window)

        # Username and password input fields
        self.usernameInput = self.findChild(QtWidgets.QLineEdit, 'USERNAME')
        self.passwordInput = self.findChild(QtWidgets.QLineEdit, 'pass')
        self.passwordInput.setEchoMode(QtWidgets.QLineEdit.Password)

    def login(self):
        username = self.usernameInput.text().strip()
        password = self.passwordInput.text().strip()

        try:
            df = pd.read_excel('user.xlsx')
            df['username'] = df['username'].astype(str).str.strip()
            df['password'] = df['password'].astype(str).str.strip()

            if not df[(df['username'] == username) & (df['password'] == password)].empty:
                QtWidgets.QMessageBox.information(self, "Login", "Login successful!")
                self.open_home_page()
            else:
                QtWidgets.QMessageBox.warning(self, "Login", "Invalid username or password.")
        except FileNotFoundError:
            QtWidgets.QMessageBox.critical(self, "Error", "Excel file 'user.xlsx' not found.")
        except PermissionError:
            QtWidgets.QMessageBox.critical(self, "Error", "Permission denied. Close the Excel file if open.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Unexpected error: {e}")

    def open_signup_window(self):
        self.signup_window = SignUpApp(self)
        self.signup_window.show()

    def open_home_page(self):
        self.home_window = HomePage()
        self.home_window.show()
        self.close()  # Close the login window


class SignUpApp(QtWidgets.QMainWindow):
    def __init__(self, login_app):
        super(SignUpApp, self).__init__()
        self.login_app = login_app
        uic.loadUi('Sign_up.ui', self)

        # Email, username, and password fields
        self.emailInput = self.findChild(QtWidgets.QLineEdit, 'email')
        self.usernameInput = self.findChild(QtWidgets.QLineEdit, 'username')
        self.passwordInput = self.findChild(QtWidgets.QLineEdit, 'password')
        self.reenterPasswordInput = self.findChild(QtWidgets.QLineEdit, 'reenetrpassword')
        self.passwordInput.setEchoMode(QtWidgets.QLineEdit.Password)
        self.reenterPasswordInput.setEchoMode(QtWidgets.QLineEdit.Password)

        self.signupButton = self.findChild(QtWidgets.QPushButton, 'bsignup')
        self.signupButton.clicked.connect(self.signup)

    def signup(self):
        email = self.emailInput.text().strip()
        username = self.usernameInput.text().strip()
        password = self.passwordInput.text().strip()
        reenter_password = self.reenterPasswordInput.text().strip()

        if password != reenter_password:
            QtWidgets.QMessageBox.warning(self, "Error", "Passwords do not match!")
            return

        try:
            try:
                df = pd.read_excel('user.xlsx')
            except FileNotFoundError:
                # Add autism_type column
                df = pd.DataFrame(columns=['email', 'username', 'password', 'autism_type'])

            # Check if the username or email already exists
            if (df['username'] == username).any() or (df['email'] == email).any():
                QtWidgets.QMessageBox.warning(self, "Error", "Username or email already exists!")
                return

            # Add new user with an empty autism_type initially
            new_user = pd.DataFrame({
                'email': [email],
                'username': [username],
                'password': [password],
                'autism_type': [None]  # Placeholder for future autism type prediction
            })
            df = pd.concat([df, new_user], ignore_index=True)
            df.to_excel('user.xlsx', index=False)

            QtWidgets.QMessageBox.information(self, "Success", "Sign up successful!")
            self.open_cv_stage()
            self.close()

        except PermissionError:
            QtWidgets.QMessageBox.critical(self, "Error", "Permission denied. Close the Excel file if open.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Could not save to the Excel file: {e}")


    def open_cv_stage(self):
        self.cv_stage_window = CVStage1PredictionApp()
        self.cv_stage_window.show()


class HomePage(QtWidgets.QMainWindow):
    def __init__(self):
        super(HomePage, self).__init__()
        uic.loadUi('HOME.ui', self)

        # Connect buttons to their respective functions
        self.LOGOUT = self.findChild(QtWidgets.QPushButton, 'LOGOUT')
        self.LOGOUT.clicked.connect(self.logout)

        self.ACTIVATEMODE = self.findChild(QtWidgets.QPushButton, 'ACTIVATEMODE')
        self.ACTIVATEMODE.clicked.connect(self.open_activities_ui)

        self.GUIDEMODE = self.findChild(QtWidgets.QPushButton, 'GUIDEMODE')
        self.GUIDEMODE.clicked.connect(self.open_guide_mode)

    def logout(self):
        QtWidgets.QMessageBox.information(self, "Logout", "You have been logged out.")
        self.close()  # Close the home page

    def open_activities_ui(self):
        self.activities_window = ActivitiesUI()
        self.activities_window.show()
        self.close()  # Close home page

    def open_guide_mode(self):
        self.guide_window = GuideModeUI()
        self.guide_window.show()
        self.close()  # Close home page


#################################################################





# Define autism class labels
autism_classes = {
    0: 'Asperger’s Syndrome',
    1: 'Classic Autism',
    2: 'PDD-NOS (Pervasive Developmental Disorder)',
    3: 'Rett Syndrome',
}

# Define the 3D ResNet model architecture
class ResNet3D(nn.Module):
    def __init__(self, num_classes):
        super(ResNet3D, self).__init__()
        self.conv1 = nn.Conv3d(3, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool3d((1, 2, 2))
        self.res_block1 = self.make_res_block(32)
        self.res_block2 = self.make_res_block(32)
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(32, num_classes)

    def make_res_block(self, channels):
        layers = []
        for _ in range(2):
            layers.append(nn.Conv3d(channels, channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm3d(channels))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(self.relu(x))
        x = x + self.res_block1(x)
        x = x + self.res_block2(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class CVStage1PredictionApp(QtWidgets.QMainWindow):
    def __init__(self):
        super(CVStage1PredictionApp, self).__init__()
        uic.loadUi('STAGE1CV.ui', self)

        self.textBrowser = self.findChild(QtWidgets.QTextBrowser, 'textBrowser')
        self.VIDEOUPLODE = self.findChild(QtWidgets.QPushButton, 'VIDEOUPLODE')
        self.NEXT = self.findChild(QtWidgets.QPushButton, 'NEXT')

        self.VIDEOUPLODE.clicked.connect(self.upload_video)
        self.NEXT.clicked.connect(self.next_stage)

        self.model_path = 'autism_model.pth'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cv_model = ResNet3D(num_classes=len(autism_classes))
        self.cv_model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.cv_model.to(self.device)
        self.cv_model.eval()

        self.video_path = None
        self.cv_prediction = None
        self.cv_confidence = None

    def upload_video(self):
        self.video_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Upload Video", "", "Video Files (*.mp4)")
        if self.video_path:
            self.textBrowser.append(f"Video uploaded: {self.video_path}")
            self.predict_autism_type()

    def preprocess_video(self, video_path, n_frames=16):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while len(frames) < n_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (256, 256))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        frames = np.array(frames)
        frames_normalized = frames / 255.0
        return frames_normalized

    def predict_autism_type(self):
        if not self.video_path:
            self.textBrowser.append("Error: No video file uploaded.")
            return

        video_frames = self.preprocess_video(self.video_path)
        if video_frames.shape[0] < 16:
            self.textBrowser.append("Error: Video is too short. Please upload a longer video.")
            return

        video_tensor = torch.tensor(video_frames, dtype=torch.float32).permute(3, 0, 1, 2).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.cv_model(video_tensor)
            yhat = nn.Softmax(dim=1)(output)
            predicted_class = torch.argmax(yhat, dim=1).item()
            confidence = torch.max(yhat).item()

        predicted_label = autism_classes.get(predicted_class, "Unknown class")
        self.cv_prediction = predicted_label
        self.cv_confidence = confidence
        self.textBrowser.append(f"Predicted Autism Type: {predicted_label}")

    def next_stage(self):
        # Pass only CV prediction and confidence to the NLP stage
        self.nlp_stage_window = NLPStage1PredictionApp(cv_prediction=self.cv_prediction, cv_confidence=self.cv_confidence)
        self.nlp_stage_window.show()
        self.close()

###########################################################


class NLPStage1PredictionApp(QtWidgets.QMainWindow):
    def __init__(self, cv_prediction=None, cv_confidence=None):
        super(NLPStage1PredictionApp, self).__init__()
        uic.loadUi('STAGE1NLP.ui', self)

        # UI Components
        self.answerbox = self.findChild(QtWidgets.QLineEdit, 'answerbox')
        self.QUATIONBOX = self.findChild(QtWidgets.QTextBrowser, 'QUATIONBOX')
        self.NEXT = self.findChild(QtWidgets.QPushButton, 'NEXT')

        # Model Setup
        self.model_path = 'trained_models'
        self.model = BertForSequenceClassification.from_pretrained(self.model_path)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        self.label_encoder = joblib.load('trained_models/label_encoder.pkl')

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

        # Questions and Data
        self.questions_df = pd.read_csv('autism_questions.csv')
        self.current_question_index = 0
        self.answers = []
        self.cv_prediction = cv_prediction
        self.cv_confidence = cv_confidence

        # Define Classes
        self.cv_classes = ['Asperger’s Syndrome', 'Classic Autism', 'PDD-NOS', 'Rett Syndrome']
        self.nlp_classes = ['Asperger’s Syndrome', 'Classic Autism', 'PDD-NOS', 'Rett Syndrome', 'High-Functioning Autism', 'Childhood Disintegrative Disorder']  # 6 NLP classes

        # Update the first question
        self.update_question()

        # Connect Button
        self.NEXT.clicked.connect(self.next_question)

    def update_question(self):
        if self.current_question_index < len(self.questions_df):
            question = self.questions_df.iloc[self.current_question_index]['Question']
            self.QUATIONBOX.setPlainText(question)
            self.answerbox.clear()
        else:
            self.predict_autism_type()
            self.go_to_home_page()

    def next_question(self):
        answer = self.answerbox.text().strip()
        self.answers.append(answer)
        self.current_question_index += 1
        self.update_question()

    def predict_autism_type(self):
        # Initialize probabilities for both models
        probabilities = []

        # Process NLP answers and calculate probabilities
        for answer in self.answers:
            inputs = self.tokenizer(answer, return_tensors='pt', padding=True, truncation=True, max_length=128).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
            probabilities.append(probs.cpu().numpy())

        # Convert probabilities to numpy array and compute average for NLP model
        probabilities = np.array(probabilities)
        avg_nlp_probabilities = np.mean(probabilities, axis=0)

        # Prepare CV model's output (zero-padded to match NLP class structure)
        cv_probabilities = np.zeros(6)  # Initialize with zeros to have 6 values for matching NLP classes

        # Check if CV prediction is available and map it correctly
        if self.cv_prediction:
            try:
                # Ensure that CV prediction is within the valid classes
                cv_class_index = self.cv_classes.index(self.cv_prediction)
                cv_probabilities[cv_class_index] = self.cv_confidence
            except ValueError:
                print(f"CV Prediction '{self.cv_prediction}' not found in CV classes. Skipping CV prediction.")

        # Now determine the final prediction
        if self.cv_prediction and self.cv_prediction in self.nlp_classes:
            # If CV and NLP predictions match, use either of them as the final prediction
            final_predicted_class = self.cv_prediction
        else:
            # Otherwise, use NLP's highest probability
            final_predicted_class = self.nlp_classes[np.argmax(avg_nlp_probabilities)]

        # Show final prediction
        QtWidgets.QMessageBox.information(self, "Final Prediction", 
                                        f"Final Prediction: {final_predicted_class}")

        # Save the final label to a file (append mode to preserve existing content)
        with open('alpha_type.txt', 'w') as file:
            file.write(f"{final_predicted_class}\n")


    def go_to_home_page(self):
        QtWidgets.QMessageBox.information(self, "Information", "All questions answered. Returning to home page.")
        self.close()

        
class ActivitiesUI(QtWidgets.QMainWindow):
    def __init__(self):
        super(ActivitiesUI, self).__init__()
        uic.loadUi('ACTIVATE.ui', self)

        # Initialize image index
        self.current_image_index = 0
        self.images = []

        # Load images and set up UI elements
        self.image_label = self.findChild(QtWidgets.QLabel, 'image_label')
        self.load_activities()

        # Connect buttons to their respective functions
        self.UPLODEVIDEO = self.findChild(QtWidgets.QPushButton, 'UPLODEVIDEO')
        self.BACKTOHOME = self.findChild(QtWidgets.QPushButton, 'BACKTOHOME')
        self.AI_BUTTON = self.findChild(QtWidgets.QPushButton, 'AI_BUTTON')

        if self.UPLODEVIDEO:
            self.UPLODEVIDEO.clicked.connect(self.upload_video)
        if self.BACKTOHOME:
            self.BACKTOHOME.clicked.connect(self.back_to_home)
        if self.AI_BUTTON:
            self.AI_BUTTON.clicked.connect(self.detect_emotion)

    def load_activities(self):
        try:
            with open('alpha_type.txt', 'r') as file:
                autism_type = file.read().strip()
        except FileNotFoundError:
            QtWidgets.QMessageBox.warning(self, "Error", "alpha_type.txt not found.")
            return

        activities_path = 'activates'
        autism_folder = os.path.join(activities_path, autism_type)
        
        if os.path.exists(autism_folder):
            self.images = [os.path.join(autism_folder, f) for f in os.listdir(autism_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
            if self.images:
                self.show_image(self.images[self.current_image_index])
            else:
                QtWidgets.QMessageBox.warning(self, "Error", f"No images found for autism type: {autism_type}")
        else:
            QtWidgets.QMessageBox.warning(self, "Error", f"No folder found for autism type: {autism_type}")

    def show_image(self, image_path):
        pixmap = QtGui.QPixmap(image_path)
        self.image_label.setPixmap(pixmap)
        self.image_label.setScaledContents(True)

    def upload_video(self):
        video_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Upload Video", "", "Video Files (*.mp4 *.avi *.mov)")
        if video_path:
            self.video_path = video_path
        else:
            QtWidgets.QMessageBox.warning(self, "Error", "No video selected")

    def detect_emotion(self):
        if not hasattr(self, 'video_path'):
            QtWidgets.QMessageBox.warning(self, "Error", "No video uploaded")
            return

        # Load and preprocess the video
        video_path = self.video_path
        video_frames = self.preprocess_video(video_path)

        # Normalize and add a batch dimension
        video_frames_normalized = video_frames / 255.0

        # Load the saved emotion detection model
        model = load_model('resnet3d_model.h5')

        # Perform predictions
        yhat = model.predict(np.expand_dims(video_frames_normalized, axis=0))

        confidence_threshold = 0.8
        predicted_class = np.argmax(yhat, axis=1)[0]
        confidence = np.max(yhat, axis=1)[0]

        if confidence >= confidence_threshold:
            if predicted_class == 1:
                emotion = 'Angry'
            elif predicted_class == 0:
                emotion = 'Happy'
            else:
                emotion = 'Sad'
          
            print(f"Predicted emotion: {emotion} with confidence: {confidence:.2f}")
            if emotion == 'Happy':
                self.load_next_activity()
            else:
                QtWidgets.QMessageBox.information(self, "Emotion Detected", f"Emotion detected: {emotion} with confidence {confidence:.2f}. Continue with the current activity.")
        else:
            QtWidgets.QMessageBox.information(self, "Low Confidence", "The model is not confident enough to determine the emotion. Please try again or provide another video.")

    def preprocess_video(self, video_path, n_frames=16):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while len(frames) < n_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (256, 256))
            frames.append(frame)
        cap.release()
        
        while len(frames) < n_frames:
            frames.append(frames[-1])
        return np.array(frames)

    def load_next_activity(self):
        if self.images:
            self.current_image_index = (self.current_image_index + 1) % len(self.images)
            self.show_image(self.images[self.current_image_index])
        else:
            QtWidgets.QMessageBox.information(self, "End of Activities", "No more activities available.")

    def back_to_home(self):
        self.home_page = HomePage()
        self.home_page.show()
        self.close()


class GuideModeUI(QtWidgets.QMainWindow):
    def __init__(self):
        super(GuideModeUI, self).__init__()
        uic.loadUi('GUIDEMODE.ui', self)  # Load the UI file

        # UI Elements
        self.HOMEPAGEBUTTON = self.findChild(QtWidgets.QPushButton, 'HOMEPAGEBUTTON')
        self.LISTSHOWIER = self.findChild(QtWidgets.QTextBrowser, 'LISTSHOWIER')
        self.resultTable = QtWidgets.QTableWidget(self)  # Initialize the result table manually

        # Configure result table
        self.resultTable.setColumnCount(3)
        self.resultTable.setHorizontalHeaderLabels(["Doctor Name", "Specialty", "Address"])

        # Button connections
        self.HOMEPAGEBUTTON.clicked.connect(self.go_to_home_page)

        # Organize UI Layout
        self.setup_ui_layout()

        # Display simulated data for autism-related specialists
        self.display_simulated_specialists()

    def go_to_home_page(self):
        # Navigate to the HomePage (ensure HomePage class is defined)
        self.home_page = HomePage()
        self.home_page.show()
        self.close()  # Close the current window when going to home page

    def display_simulated_specialists(self):
        # Simulated list of autism-related doctors or specialists
        specialist_data = [
            {"name": "Dr. John Smith", "specialty": "Autism Specialist", "address": "123 Main St, Anytown"},
            {"name": "Dr. Emily Johnson", "specialty": "Child Psychologist", "address": "456 Oak Ave, Sampletown"},
            {"name": "Dr. Sarah Brown", "specialty": "Behavioral Therapist", "address": "789 Pine Rd, Example City"}
        ]

        # Fill the table with simulated data
        self.resultTable.setRowCount(len(specialist_data))
        for row, specialist in enumerate(specialist_data):
            self.resultTable.setItem(row, 0, QtWidgets.QTableWidgetItem(specialist["name"]))
            self.resultTable.setItem(row, 1, QtWidgets.QTableWidgetItem(specialist["specialty"]))
            self.resultTable.setItem(row, 2, QtWidgets.QTableWidgetItem(specialist["address"]))

    def setup_ui_layout(self):
        # Setting up the layout for better organization
        layout = QtWidgets.QVBoxLayout()

        # Add result table to the layout
        layout.addWidget(self.resultTable)

        # Add list text browser (LISTSHOWIER) to the layout
        layout.addWidget(self.LISTSHOWIER)

        # Spacer to push the button to the bottom
        spacer = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        layout.addItem(spacer)

        # Add Back to Home button to the layout
        layout.addWidget(self.HOMEPAGEBUTTON)

        # Set up the central widget with the layout
        central_widget = QtWidgets.QWidget(self)
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def back_to_home(self):
        self.home_page = HomePage()
        self.home_page.show()
        self.close()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = LoginApp()
    window.show()
    sys.exit(app.exec_())

