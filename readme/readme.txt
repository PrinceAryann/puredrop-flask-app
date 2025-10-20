PureDrop: AI-Powered Water Quality Analyzer

An intuitive web application built with Flask and PyTorch that predicts water potability based on standard chemical analysis. Users can input values from a water report to receive an instant prediction on whether the water is safe to drink and a breakdown of potential health risks if contaminants exceed safe levels.

Live Demo: [Link to your hosted application on Render]

Key Features

Instant Prediction: Get an immediate "Safe" or "Not Safe" result powered by a pre-trained neural network.

Health Risk Analysis: Automatically identifies which parameters exceed safe limits and lists the associated health concerns.

User-Friendly Interface: A clean, responsive form makes entering water quality data simple and straightforward on any device.

Full-Stack Application: Demonstrates a complete workflow from a Python backend and machine learning model to a dynamic frontend.

Screenshots

<table align="center">
<tr>
<td align="center"><b>Homepage</b></td>
<td align="center"><b>Analysis & Results Page</b></td>
</tr>
<tr>
<td><img src="https://www.google.com/search?q=https://i.imgur.com/your-homepage-screenshot.png" alt="Homepage Screenshot" width="400"/></td>
<td><img src="https://www.google.com/search?q=https://i.imgur.com/your-analysis-screenshot.png" alt="Analysis Page Screenshot" width="400"/></td>
</tr>
</table>

(To add screenshots, take pictures of your running application, upload them to a service like Imgur, and paste the image links here.)

Technology Stack

Backend: Python, Flask

Machine Learning: PyTorch, NumPy

Frontend: HTML, CSS, Bootstrap 5

Deployment: Gunicorn

Local Setup & Installation

To run this project on your local machine, follow these steps:

Clone the repository:

git clone [https://github.com/YourUsername/your-repo-name.git](https://github.com/YourUsername/your-repo-name.git)
cd your-repo-name


Create and activate a virtual environment:

# Using Conda
conda create --name puredrop-env python=3.10
conda activate puredrop-env


Install the required dependencies:

pip install -r requirements.txt


Run the application:

python main.py


The application will be available at http://127.0.0.1:5000.