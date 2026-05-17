# PSBD: A Two-Phase Defense Pipeline against Neural Network Backdoors

Reference implementation accompanying the paper:

> **A Two-Phase Defense Pipeline against Neural Network Backdoors: Integrating
> Prediction Shift Uncertainty and Secure Recovery.**
> Suraj Yadav, Department of Computer Science, Delhi Technological University.

The repository implements a **two-phase backdoor defense** evaluated on
CIFAR-10 against three attacks of increasing stealth:

1. **BadNets** — localized 4×4 white patch trigger.
2. **Blended** — global alpha-blended noise trigger.
3. **WaNets** — imperceptible elastic warping trigger.

The defense itself has two phases:

- **Phase 1 — PSBD filter.** Train an infected baseline ResNet-18, build a
  weight-shared copy with `TestTimeDropout` layers, and use Prediction Shift
  Uncertainty (PSU) to flag samples whose prediction confidence collapses
  under architectural perturbation.
- **Phase 2 — Secure recovery.** Train a *fresh* ResNet-18 from scratch on
  the verified clean pool only, with heavy augmentation and a
  `ReduceLROnPlateau` schedule.

---

## Repository layout

```
psbd-backdoor-defense/
├── main.py                              # CLI entry point (full pipeline)
├── requirements.txt
├── LICENSE
├── src/
│   ├── config.py                        # All hyperparameters
│   ├── data_loader.py                   # Native CIFAR-10 downloader
│   ├── attacks/
│   │   ├── badnets.py                   # Section IV.3 (1)
│   │   ├── blended.py                   # Section IV.3 (2)
│   │   ├── wanets.py                    # Section IV.3 (3)
│   │   └── injector.py                  # Dispatch + poison injection
│   ├── models/
│   │   └── resnet18.py                  # ResNet-18 + TestTimeDropout
│   ├── defense/
│   │   └── psbd_filter.py               # PSU computation + thresholding
│   ├── training/
│   │   ├── phase1_infected_model.py     # Train infected baseline
│   │   └── phase2_secure_recovery.py    # Train secure model
│   └── evaluation/
│       └── metrics.py                   # TPR / FPR / CSR / ASR
├── scripts/
│   ├── run_all_attacks.ps1              # Windows convenience runner
│   └── run_all_attacks.sh               # Linux / macOS convenience runner
└── results/                             # JSON metric reports land here
```

Each Python module maps cleanly onto a section of the paper, so a reviewer
can jump directly from an equation or table to the code that produces it.

---

## Quick start

### 1. Install dependencies

The code targets Python 3.9 – 3.11 and TensorFlow 2.10 – 2.15.

```bash
git clone https://github.com/<your-username>/psbd-backdoor-defense.git
cd psbd-backdoor-defense

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

> **GPU users:** install the TensorFlow build that matches your CUDA / cuDNN
> toolkit (`pip install "tensorflow[and-cuda]"` on Linux). The pipeline runs
> on CPU too but Phase 1 (55 epochs) is meaningfully slower.

### 2. Run a single experiment

```bash
python main.py --attack wanets
```

Replace `wanets` with `badnets` or `blended` to reproduce the other rows of
Tables 1 and 2.

### 3. Run all three attacks back-to-back

Linux / macOS:

```bash
bash scripts/run_all_attacks.sh
```

Windows (PowerShell):

```powershell
.\scripts\run_all_attacks.ps1
```

After each run a JSON report is dropped in `./results/` with the PSBD filter
summary plus final CSR / ASR — convenient for plotting or for an appendix
table.

---

## Command-line options

```
python main.py [--attack {badnets,blended,wanets}]
               [--poison-ratio FLOAT]
               [--target-label INT]
               [--seed INT]
               [--phase1-epochs INT]
               [--phase2-epochs INT]
               [--dropout-rate FLOAT]
               [--results-dir PATH]
```

| Flag                | Default   | Meaning                                                                 |
| ------------------- | --------- | ----------------------------------------------------------------------- |
| `--attack`          | `wanets`  | Which trigger to inject (Section IV.3).                                 |
| `--poison-ratio`    | `0.10`    | Fraction of `D_train` the attacker poisons (Section IV.2).              |
| `--target-label`    | `0`       | Attacker's target class — `0` is *airplane*.                            |
| `--phase1-epochs`   | `55`      | Epochs used to train the infected baseline.                             |
| `--phase2-epochs`   | `80`      | Epochs used to train the final secure model (Section III.3.2).          |
| `--dropout-rate`    | `0.6`     | Probability used in `TestTimeDropout` during the PSBD scan.             |
| `--seed`            | `42`      | Seeds Python / NumPy / TensorFlow for reproducibility.                  |
| `--results-dir`     | `results` | Where the JSON metrics file is written.                                 |

To explore the ablation in Section VI.2 (sensitivity to test-time dropout),
pass `--dropout-rate 0.2`, `0.5`, `0.6`, or `0.8`. For the poison-ratio
ablation in Section VI.1, vary `--poison-ratio` between `0.01` and `0.20`.

---

## License

Released under the MIT License. See [LICENSE](LICENSE).
