1) create and activate venv
python -m venv .venv
.venv\Scripts\Activate.ps1  (Windows PowerShell)
# or .venv\Scripts\activate.bat (cmd)

2) install backend deps
pip install -r backend/requirements.txt

3) prepare crops (one-time, or if you already have labels)
python backend/prepare_crops.py --images .\affordance_dataset\images\train --labels .\affordance_dataset\labels\train --out .\affordance_crops --classnames .\affordance_dataset\classes.txt

4) train affordance classifier (adjust epochs)
python backend/train_affordance.py --csv .\affordance_crops\dataset.csv --out .\backend_models\affordance_clf.pt --epochs 10 --batch 16

5) run server
# make sure yo lov8n weights are available (ultralytics will auto-download)
python backend/server.py

6) serve frontend
cd frontend
python -m http.server 3000
Open http://localhost:3000
