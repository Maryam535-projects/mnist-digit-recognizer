# 🤖 MNIST Handwritten Digit Recognizer

A **web-based AI application** that recognizes handwritten digits (0-9) with **~99.5% accuracy**. Users can draw digits or upload images for instant predictions.

---

## **Features**

### 🎨 User Interface

* Interactive drawing canvas with **touch support**
* Image upload (PNG, JPG, JPEG)
* Real-time predictions with **confidence scores**
* Visual probability distribution for all digits
* Responsive design for **all devices**

### 🧠 AI Capabilities

* Achieves **~99.5% accuracy** on MNIST test set
* **CNN architecture** optimized for handwritten digits
* Special handling for **challenging digits** (e.g., 9)
* One-time training, **instant model loading** on subsequent runs
* Inference time **<100ms**

### 🛠️ Technical Features

* Model persistence — trained model saved locally
* Comprehensive error handling
* Production-ready **Flask backend**
* Modern web technologies (HTML, CSS, JS)
* Clean, modular, and maintainable codebase

---

## **🚀 Quick Start**

### Prerequisites

* **Python 3.10+**
* pip package manager

### Installation

```bash
# Clone the project
git clone https://github.com/Maryam535-projects/mnist-digit-recognizer.git
cd mnist-digit-recognizer

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

Open in browser: [http://localhost:5000](http://localhost:5000)

---

### **First Run**

* Downloads MNIST dataset (70,000 images)
* Trains CNN model (~4-5 minutes)
* Saves trained model locally
* Starts web server

### **Subsequent Runs**

* Loads pre-trained model instantly
* Ready for predictions in **<1 second**

---

## **🛠️ Technology Stack**

**Backend**

* Python 3.10+ with **Flask**
* **PyTorch 2.9.1** for deep learning
* NumPy & Pillow for image processing

**Frontend**

* HTML5 Canvas for drawing
* CSS3 with modern styling
* JavaScript for interactivity

**AI/ML**

* CNN Architecture with 8 layers
* MNIST dataset: 70,000 handwritten digits
* Model persistence (saved weights for instant loading)

---

## **📁 Project Structure**

```
mnist-digit-recognizer/
├── app.py                    # Main Flask application
├── mnist_perfect_model.pt    # Trained model (after first run)
├── model_info.json           # Model metadata
├── requirements.txt          # Dependencies
├── templates/
│   └── index.html            # Web interface
└── data/                     # MNIST dataset
```

---

## **🎮 How to Use**

### 1️⃣ Drawing Digits

1. Draw any digit (0-9) on the canvas
2. Click **"Predict Drawing"**
3. View prediction with confidence score
4. See probability distribution for all digits

### 2️⃣ Uploading Images

1. Click **"Upload Image"**
2. Select PNG/JPG/JPEG file
3. Get instant prediction
4. Download detailed report (optional)

### 3️⃣ Testing All Digits

* Draw digits 0-9 to verify accuracy
* Each should show **>98% confidence**
* Special handling ensures challenging digits (like 9) are predicted reliably

---

## **🧠 Model Architecture**

**CNN Layers**

```
Input (28×28) 
→ Conv2D(32) → BatchNorm → ReLU → MaxPool
→ Conv2D(64) → BatchNorm → ReLU → MaxPool
→ Conv2D(128) → BatchNorm → ReLU → MaxPool
→ Flatten → Dense(256) → Dropout
→ Dense(128) → Dropout
→ Dense(10) → Softmax → Output
```

**Training Details**

* **Dataset**: MNIST 70,000 images
* **Epochs**: 8 with early stopping
* **Accuracy**: ~99.5% on test set
* **Training time**: 4-5 minutes (first run only)

**Performance Metrics**

| Metric           | Value       |
| ---------------- | ----------- |
| Overall Accuracy | ~99.53%     |
| Inference Time   | <100ms      |
| Model Size       | ~5MB        |
| Training Time    | 4-5 minutes |
| Loading Time     | <1 second   |

---

## **🔧 Development**

### Running Tests

```bash
# Check if PyTorch is installed
python -c "import torch; print('PyTorch available')"

# Check Flask
python -c "import flask; print('Flask available')"
```

### File Descriptions

* **app.py**: Main application with backend logic
* **index.html**: Web interface with drawing canvas
* **requirements.txt**: Python dependencies

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

**Guidelines**

* Follow **PEP 8** style guide
* Add comments for complex logic
* Test all changes thoroughly
* Update documentation if needed

---

## **📄 License**

This project is licensed under the **MIT License** — see LICENSE file for details.

---

## **👨‍💻 Author**

**Maryam Shuaib**

* GitHub: [@Maryam535-projects](https://github.com/Maryam535-projects)
* Email: [maryamshuaib934@gmail.com](mailto:maryamshuaib934@gmail.com)

---

## **🙏 Acknowledgments**

* MNIST dataset creators
* PyTorch and Flask communities
* All contributors and testers

---

## **💡 Pro Tips**

1. Draw digits clearly and centered
2. Use thick lines on the canvas for best results
3. Test with all digits 0-9 to verify accuracy
4. First run trains the model — subsequent runs are instant

---

## **🚀 Quick Commands**

```bash
# Run the app
python app.py

# Force retrain (delete model file)
del mnist_perfect_model.pt
```

**Enjoy recognizing digits with AI!** 🎯
