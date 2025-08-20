import sys
import pandas as pd
from PyQt5 import QtWidgets, uic
import os

class LoginApp(QtWidgets.QMainWindow):
    def __init__(self):
        super(LoginApp, self).__init__()
        
        # Load the UI file
        ui_file_path = os.path.join(os.path.dirname(__file__), 'Login.ui')
        uic.loadUi(ui_file_path, self)

        # Connect the login and sign-up buttons to respective functions
        self.loginButton = self.findChild(QtWidgets.QPushButton, 'BLOGIN')
        self.signupButton = self.findChild(QtWidgets.QPushButton, 'BSIGNUP')

        self.loginButton.clicked.connect(self.login)
        self.signupButton.clicked.connect(self.open_signup_window)

        # Username and password input fields
        self.usernameInput = self.findChild(QtWidgets.QLineEdit, 'USERNAME')
        self.passwordInput = self.findChild(QtWidgets.QLineEdit, 'PASSWORD')

        # Debug output
        print(f'loginButton: {self.loginButton}')
        print(f'signupButton: {self.signupButton}')
        print(f'usernameInput: {self.usernameInput}')
        print(f'passwordInput: {self.passwordInput}')

        if self.passwordInput is not None:
            self.passwordInput.setEchoMode(QtWidgets.QLineEdit.Password)
        else:
            print("Error: passwordInput is None, check your UI file.")

    def login(self):
        username = self.usernameInput.text().strip()
        password = self.passwordInput.text().strip()

        # Read the Excel file
        try:
            df = pd.read_excel('user.xlsx')  # Ensure this path matches your file location

            # Convert columns to string to avoid type errors
            df['username'] = df['username'].astype(str).str.strip()
            df['password'] = df['password'].astype(str).str.strip()

            # Check if the username and password exist in the DataFrame
            if not df[(df['username'] == username) & (df['password'] == password)].empty:
                print("Login successful!")
            else:
                print("Invalid username or password.")
        except Exception as e:
            print(f"Error reading the Excel file: {e}")

    def open_signup_window(self):
        # Open the sign-up window
        self.signup_window = SignUpApp()
        self.signup_window.show()

class SignUpApp(QtWidgets.QWidget):
    def __init__(self):
        super(SignUpApp, self).__init__()
        uic.loadUi('Sign_up.ui', self)  # Ensure the path to your Sign_up.ui file is correct

        # Email, username, and password fields
        self.emailInput = self.findChild(QtWidgets.QLineEdit, 'email')
        self.usernameInput = self.findChild(QtWidgets.QLineEdit, 'username')
        self.passwordInput = self.findChild(QtWidgets.QLineEdit, 'password')
        self.reenterPasswordInput = self.findChild(QtWidgets.QLineEdit, 'reenetrpassword')

        # Connect the sign-up button to the sign-up function
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

        # Read the existing file or create a new DataFrame
        try:
            df = pd.read_excel('user.xlsx')
        except FileNotFoundError:
            df = pd.DataFrame(columns=['email', 'username', 'password'])

        # Check if username or email already exists
        if (df['username'] == username).any() or (df['email'] == email).any():
            QtWidgets.QMessageBox.warning(self, "Error", "Username or email already exists!")
            return

        # Add new user data and save it to the Excel file
        new_user = pd.DataFrame({'email': [email], 'username': [username], 'password': [password]})
        df = pd.concat([df, new_user], ignore_index=True)

        try:
            df.to_excel('user.xlsx', index=False)
            QtWidgets.QMessageBox.information(self, "Success", "Sign up successful!")
            self.close()  # Close the sign-up window after successful sign-up
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Could not save to the Excel file: {e}")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = LoginApp()
    window.show()
    sys.exit(app.exec_())
