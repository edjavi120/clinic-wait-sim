# Clinic Wait-Time Simulator

An interactive health data exhibit (CAI5735, Module 14 paired exercise).
A small neural network predicts how long a non-urgent walk-in patient will wait at a clinic,
based on staffing and the queue. Visitors adjust the clinic and watch the prediction change.
A WebGPU chatbot (WebLLM) answers questions from the site guide.

**Question:** How do staffing and queue size affect wait times?
**Task:** Regression · **Data:** synthetic (discrete-event queue simulation, no real patients)

## Files
| Path | What it is |
|---|---|
| `train_model.py` | Simulates 8,000 clinic arrivals, trains the network, compares it with baselines, finds the biggest mistake, exports `docs/model.json` |
| `data/clinic_waits.csv` | The generated dataset |
| `docs/index.html` | The exhibit (runs the trained network in the browser) |
| `docs/model.json` | Trained weights, scaler, test results |

## Model
- **Inputs:** doctors on shift (1–6), patients ahead (0–15), avg visit length (5–25 min), walk-ins/hour (2–15)
- **Output:** predicted wait (minutes)
- **Network:** scikit-learn MLPRegressor, 1 hidden layer of 32 ReLU units, trained on log(1 + wait). Size picked on a validation set.
- **Split:** 70% train / 15% validation / 15% test

## Test results (1,200 held-out patients)
| Model | MAE (min) | R² |
|---|---|---|
| Neural network | 11.7 | 0.903 |
| Baseline: median training wait (35.2 min) | 47.9 | −0.122 |
| Linear regression | 32.1 | 0.572 |

**Biggest mistake:** 1 doctor, 14 ahead, 20.7-min visits, 12.4 walk-ins/hr. Predicted 8.4 hr, actual 15.0 hr.
500 reruns of that same clinic had a median of 8.2 hr, so the model got the typical wait right and that patient was unlucky.
Predicting a range, or knowing how many urgent patients were on the way, would have helped.

## Run it
```bash
pip install -r requirements.txt
python train_model.py          # regenerates data/ and docs/model.json
python -m http.server -d docs  # open http://localhost:8000
```

## Deploy (GitHub Pages)
Settings → Pages → Source: *Deploy from a branch* → `main` / `/docs`.

Chat needs a WebGPU browser (recent Chrome/Edge on desktop). The rest of the exhibit works without it.
