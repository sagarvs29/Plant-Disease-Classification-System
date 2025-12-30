# 🌿 Plant Disease Classification System

A deep learning-based web application for detecting and classifying plant diseases from leaf images using Convolutional Neural Networks (CNN). This system can identify 38 different plant disease classes across various crops including tomatoes, potatoes, apples, grapes, and more.

## 📋 Table of Contents
- [Features](#-features)
- [Dataset](#-dataset)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
  - [Training the Model](#training-the-model)
  - [Running the Web Application](#running-the-web-application)
- [Model Architecture](#-model-architecture)
- [Supported Plant Diseases](#-supported-plant-diseases)
- [Screenshots](#-screenshots)
- [Results](#-results)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

## ✨ Features

- **Real-time Disease Detection**: Upload plant leaf images and get instant disease predictions
- **38 Disease Classes**: Supports detection of 38 different plant diseases across multiple crops
- **User-Friendly Web Interface**: Clean, modern, and intuitive web UI built with Flask
- **High Accuracy Model**: Deep CNN model trained on a comprehensive plant disease dataset
- **Image Preprocessing**: Automatic image preprocessing and normalization
- **Interactive Predictions**: Visual feedback with uploaded image display and prediction results

## 📊 Dataset

The model is trained on the **PlantVillage Dataset**, which contains:
- **Training Set**: 70,295 images
- **Validation Set**: 17,572 images
- **Test Set**: 33 images
- **Image Size**: 224x224 pixels
- **Classes**: 38 different plant disease categories
- **Format**: Categorized by plant type and disease name

## 🛠️ Tech Stack

### Machine Learning & Deep Learning
- **TensorFlow**: 2.19.0
- **Keras**: Deep learning model building and training
- **NumPy**: Numerical computations
- **Scikit-learn**: Metrics and evaluation
- **Matplotlib & Seaborn**: Visualization

### Web Framework
- **Flask**: Web application framework
- **HTML/CSS**: Frontend interface
- **JavaScript**: Client-side interactions

### Development Tools
- **Jupyter Notebook**: Model development and experimentation
- **Python**: 3.x

## 📁 Project Structure

```
Plant-Disease-Classification-System/
│
├── app.py                          # Flask web application
├── plant ml project.ipynb          # Jupyter notebook for model training
├── templates/
│   └── index.html                  # Web interface template
├── static/                         # Static files (uploaded images)
├── plant_disease_model.keras       # Trained model (generated after training)
├── .gitignore                      # Git ignore file
└── README.md                       # Project documentation
```

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/sagarvs29/Plant-Disease-Classification-System.git
cd Plant-Disease-Classification-System
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install tensorflow==2.19.0
pip install flask
pip install numpy
pip install pillow
pip install scikit-learn
pip install matplotlib
pip install seaborn
pip install jupyter
pip install ipywidgets
```

Or create a `requirements.txt` file:
```txt
tensorflow==2.19.0
flask==3.0.0
numpy==1.24.3
pillow==10.1.0
scikit-learn==1.3.2
matplotlib==3.8.2
seaborn==0.13.0
jupyter==1.0.0
ipywidgets==8.1.1
```

Then install:
```bash
pip install -r requirements.txt
```

### Step 4: Download the Dataset
Download the PlantVillage dataset and organize it into `train/`, `valid/`, and `test/` directories.

## 📖 Usage

### Training the Model

1. **Open the Jupyter Notebook**:
   ```bash
   jupyter notebook "plant ml project.ipynb"
   ```

2. **Configure Dataset Path**: Update the dataset path in the notebook to point to your dataset location:
   ```python
   dataset_path = "path/to/your/dataset"
   ```

3. **Run All Cells**: Execute all cells in the notebook to:
   - Load and preprocess the dataset
   - Build the CNN model
   - Train the model (default: 10 epochs)
   - Evaluate model performance
   - Save the trained model as `plant_disease_model.keras`

4. **Training Parameters**:
   - Batch Size: 32
   - Image Size: 224x224
   - Epochs: 10 (adjustable)
   - Optimizer: Adam
   - Loss Function: Categorical Crossentropy

### Running the Web Application

1. **Ensure Model File Exists**: Make sure `plant_disease_model.keras` is in the project root directory.

2. **Create Static Directory**:
   ```bash
   mkdir static
   ```

3. **Run the Flask App**:
   ```bash
   python app.py
   ```

4. **Access the Application**:
   - Open your web browser
   - Navigate to `http://127.0.0.1:5000/`
   - Upload a plant leaf image
   - Click "Predict" to get the disease classification

## 🧠 Model Architecture

The CNN model consists of the following layers:

```
Input Layer (224, 224, 3)
    ↓
Conv2D (32 filters, 3x3, ReLU)
    ↓
MaxPooling2D (2x2)
    ↓
Conv2D (64 filters, 3x3, ReLU)
    ↓
MaxPooling2D (2x2)
    ↓
Conv2D (128 filters, 3x3, ReLU)
    ↓
MaxPooling2D (2x2)
    ↓
Flatten
    ↓
Dropout (0.5)
    ↓
Dense (256 units, ReLU)
    ↓
Dense (38 units, Softmax)
```

**Model Parameters**:
- Total Parameters: ~22.2 Million
- Trainable Parameters: ~22.2 Million
- Activation: ReLU (hidden layers), Softmax (output layer)
- Regularization: Dropout (0.5)

## 🌱 Supported Plant Diseases

The system can classify the following 38 plant disease categories:

### Apple (4 classes)
- Apple Scab
- Black Rot
- Cedar Apple Rust
- Healthy

### Blueberry (1 class)
- Healthy

### Cherry (2 classes)
- Powdery Mildew
- Healthy

### Corn/Maize (4 classes)
- Cercospora Leaf Spot (Gray Leaf Spot)
- Common Rust
- Northern Leaf Blight
- Healthy

### Grape (4 classes)
- Black Rot
- Esca (Black Measles)
- Leaf Blight (Isariopsis Leaf Spot)
- Healthy

### Orange (1 class)
- Huanglongbing (Citrus Greening)

### Peach (2 classes)
- Bacterial Spot
- Healthy

### Pepper/Bell (2 classes)
- Bacterial Spot
- Healthy

### Potato (3 classes)
- Early Blight
- Late Blight
- Healthy

### Raspberry (1 class)
- Healthy

### Soybean (1 class)
- Healthy

### Squash (1 class)
- Powdery Mildew

### Strawberry (2 classes)
- Leaf Scorch
- Healthy

### Tomato (10 classes)
- Bacterial Spot
- Early Blight
- Late Blight
- Leaf Mold
- Septoria Leaf Spot
- Spider Mites (Two-spotted Spider Mite)
- Target Spot
- Yellow Leaf Curl Virus
- Tomato Mosaic Virus
- Healthy

## 📸 Screenshots

### Web Interface
The application features a modern, gradient-based design with:
- Clean upload interface
- Real-time image preview
- Instant prediction results
- Responsive layout

### Prediction Display
- Uploaded image visualization
- Disease classification result
- Confidence-based feedback

## 📈 Results

The model achieves:
- **Training Accuracy**: High accuracy on training set
- **Validation Accuracy**: Robust performance on validation data
- **Test Performance**: Reliable predictions on unseen data

*Note: Specific accuracy metrics will vary based on training duration and dataset quality.*

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the Repository**
2. **Create a Feature Branch**:
   ```bash
   git checkout -b feature/YourFeature
   ```
3. **Commit Your Changes**:
   ```bash
   git commit -m "Add YourFeature"
   ```
4. **Push to the Branch**:
   ```bash
   git push origin feature/YourFeature
   ```
5. **Open a Pull Request**

### Contribution Ideas
- Improve model accuracy
- Add more plant disease classes
- Enhance UI/UX design
- Add data augmentation techniques
- Implement transfer learning (ResNet, VGG, etc.)
- Add API endpoints for mobile integration
- Create Docker containerization
- Add unit tests and integration tests

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- **PlantVillage Dataset**: For providing the comprehensive plant disease image dataset
- **TensorFlow/Keras**: For the deep learning framework
- **Flask**: For the web framework
- **Contributors**: Thanks to all contributors who help improve this project

---

**Made with ❤️ for agriculture and plant health**

For issues, questions, or suggestions, please open an issue on the [GitHub repository](https://github.com/sagarvs29/Plant-Disease-Classification-System).
