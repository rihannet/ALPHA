import sys
import pandas as pd
import joblib
import numpy as np
import os
import glob
import cv2
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from tensorflow.keras.models import load_model
from PyQt5 import QtWidgets, uic


class LoginApp(QtWidgets.QMainWindow):
    def __init__(self):
        super(LoginApp, self).__init__()
        uic.loadUi('Login.ui', self)

        # Connect the login and sign-up buttons to respective functions
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
                self.open_home_page()  # Open home page on successful login
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
                df = pd.DataFrame(columns=['email', 'username', 'password'])

            if (df['username'] == username).any() or (df['email'] == email).any():
                QtWidgets.QMessageBox.warning(self, "Error", "Username or email already exists!")
                return

            new_user = pd.DataFrame({'email': [email], 'username': [username], 'password': [password]})
            df = pd.concat([df, new_user], ignore_index=True)
            df.to_excel('user.xlsx', index=False)
            QtWidgets.QMessageBox.information(self, "Success", "Sign up successful!")
            self.open_cv_stage()
            self.close()  # Close the sign-up window
        except PermissionError:
            QtWidgets.QMessageBox.critical(self, "Error", "Permission denied. Close the Excel file if open.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Could not save to the Excel file: {e}")

    def open_cv_stage(self):
        self.cv_stage_window = CVStage1PredictionApp()
        self.cv_stage_window.show()  # Open the CV Stage 1 Prediction UI


class HomePage(QtWidgets.QMainWindow):
    def __init__(self):
        super(HomePage, self).__init__()
        uic.loadUi('HOME.ui', self)

        # Connect the logout button to the logout function
        self.LOGOUT = self.findChild(QtWidgets.QPushButton, 'LOGOUT')
        self.LOGOUT.clicked.connect(self.logout)

        # Placeholder for additional buttons like ACTIVATEMODE and GUIDEMODE
        self.ACTIVATEMODE = self.findChild(QtWidgets.QPushButton, 'ACTIVATEMODE')
        self.GUIDEMODE = self.findChild(QtWidgets.QPushButton, 'GUIDEMODE')

    def logout(self):
        QtWidgets.QMessageBox.information(self, "Logout", "You have been logged out.")
        self.close()  # Close the home page


class CVStage1PredictionApp(QtWidgets.QMainWindow):
    def __init__(self):
        super(CVStage1PredictionApp, self).__init__()
        uic.loadUi('STAGE1CV.ui', self)

        # Initialize UI elements
        self.textBrowser = self.findChild(QtWidgets.QTextBrowser, 'textBrowser')
        self.VIDEOUPLODE = self.findChild(QtWidgets.QPushButton, 'VIDEOUPLODE')
        self.NEXT = self.findChild(QtWidgets.QPushButton, 'NEXT')

        self.VIDEOUPLODE.clicked.connect(self.upload_video)
        self.NEXT.clicked.connect(self.next_stage)

    def upload_video(self):
        video_file, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Upload Video", "", "Video Files (*.mp4)")
        if video_file:
            self.textBrowser.append(f"Video uploaded: {video_file}")
            # You can add functionality to process the uploaded video

    def next_stage(self):
        # Open the NLP stage upon clicking 'Next'
        self.nlp_stage_window = NLPStage1PredictionApp()
        self.nlp_stage_window.show()
        self.close()  # Close the CV stage window


class NLPStage1PredictionApp(QtWidgets.QMainWindow):
    def __init__(self):
        super(NLPStage1PredictionApp, self).__init__()
        uic.loadUi('STAGE1NLP.ui', self)

        # Initialize UI elements
        self.answerbox = self.findChild(QtWidgets.QLineEdit, 'answerbox')
        self.QUATIONBOX = self.findChild(QtWidgets.QTextBrowser, 'QUATIONBOX')
        self.NEXT = self.findChild(QtWidgets.QPushButton, 'NEXT')

        # Load questions for the NLP stage
        self.questions_df = pd.read_csv('X:/alpha/ALPHA FINEL/autism_questions.csv')
        self.current_question_index = 0
        self.update_question()

        self.NEXT.clicked.connect(self.next_question)

    def update_question(self):
        if self.current_question_index < len(self.questions_df):
            question = self.questions_df.iloc[self.current_question_index]['Question']
            self.QUATIONBOX.setPlainText(question)
            self.answerbox.clear()
        else:
            QtWidgets.QMessageBox.information(self, "Info", "You have answered all questions.")
            self.close()  # Close the window after all questions are answered

    def next_question(self):
        answer = self.answerbox.text().strip()
        # Here you can store the answer to be used in prediction later

        self.current_question_index += 1
        self.update_question()


# AI Prediction Logic
def ai_predict():
    # Set folder paths for video input
    autism_video_folder = "path/to/autism_detection"
    emotion_video_folder = "path/to/emotion_detection"
    activities_folder = "path/to/activities"  # Path to activities folder

    # Load NLP model and tokenizer
    nlp_model_path = 'path/to/autism_nlp_model'
    nlp_model = BertForSequenceClassification.from_pretrained(nlp_model_path)
    nlp_tokenizer = BertTokenizer.from_pretrained(nlp_model_path)
    label_encoder = joblib.load('path/to/label_encoder.pkl')
    nlp_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    nlp_model.to(nlp_device)

    # Load questions
    questions_df = pd.read_csv('X:/alpha/ALPHA FINEL/autism_questions.csv')
    questions = questions_df['Question'].tolist()

    # Autism CV Model (TensorFlow)
    autism_cv_model = load_model('path/to/autism_cv_model.h5')
    autism_classes = {
        2: "Classic Autism",
        1: "Asperger’s Syndrome",
        3: "PDD-NOS",
        4: "Childhood Disintegrative Disorder",
        5: "Rett Syndrome",
        6: "High-Functioning Autism",
    }

    # Process video files for predictions
    # Add your video processing code here

    # After predictions, update activity selection logic
    # Your logic for selecting activities based on predictions


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = LoginApp()
    window.show()
    sys.exit(app.exec_())
