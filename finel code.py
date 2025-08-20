import sys
import pandas as pd
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
            else:
                QtWidgets.QMessageBox.warning(self, "Login", "Invalid username or password.")
        except FileNotFoundError:
            QtWidgets.QMessageBox.critical(self, "Error", "Excel file 'user.xlsx' not found.")
        except PermissionError:
            QtWidgets.QMessageBox.critical(self, "Error", "Permission denied. Close the Excel file if open.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Unexpected error: {e}")

    def open_signup_window(self):
        self.signup_window = SignUpApp()
        self.signup_window.show()

class SignUpApp(QtWidgets.QMainWindow):
    def __init__(self):
        super(SignUpApp, self).__init__()
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
            self.close()
        except PermissionError:
            QtWidgets.QMessageBox.critical(self, "Error", "Permission denied. Close the Excel file if open.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Could not save to the Excel file: {e}")

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_window = LoginApp()
    main_window.show()
    sys.exit(app.exec_())
