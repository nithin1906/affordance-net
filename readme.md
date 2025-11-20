# AffordanceNet 👁️🧠

**AffordanceNet** is a professional-grade computer vision workstation that detects objects and identifies their functional affordances (e.g., "Graspable", "Sittable", "Containment") in real-time.

It uses a two-stage AI pipeline:
1.  **The Finder:** YOLOv8 (COCO-pretrained) to localize objects.
2.  **The Thinker:** A custom MobileNetV2 classifier to predict affordance probabilities.

![App Screenshot](https://via.placeholder.com/800x450?text=App+Screenshot+Here)

## 🚀 Quick Start (Run without Training)

You can run this project immediately using the pre-trained models included in the repo.

### Prerequisites
* **Python 3.8+**
* **Node.js & npm**
* **CUDA-capable GPU** (Recommended, but runs on CPU)

### 1. Backend Setup (The Brain)
Open a terminal in the root directory:

```bash
# 1. Create virtual environment
python -m venv .venv

# 2. Activate it
# Windows:
.venv\Scripts\Activate.ps1
# Mac/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the Server
uvicorn backend.server:app --port 8000