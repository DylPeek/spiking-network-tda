Overview
--------
Reorganized, reproducible code for three experiment tracks using LIF spiking neural networks (SNNs) and time-series TDA (Transfer Entropy → Persistent Homology → Betti/AUBC):
1) Logic gates (AND/OR/XOR),
2) MNIST (flip/move noise variants),
3) IBL Neuropixels (real-world).

Layout
------
common/
  time-series-tda/           # TE/PH/Betti helpers (imports via sys.path)
projects/
  logic_gates/
    0-and_or_xor_snn.py
    1-evaluation_tda_analysis.py
    2-evaluation_te_visual.py
    3-learning_tda_analysis.py
  mnist/
    0a-mnist_snn.py          # baseline
    0b-mnist_snn_dropout.py  # trains across flip-noise levels
    0c-mnist_snn_move.py     # trains across move-noise levels
    0d-mnist_noise_visual.py # visual examples
    1-evaluation_tda_analysis.py
    2-evaluation_te_visual.py
    3-learning_tda_analysis.py
  mouse_ibl/
    0-data_source.py
    1-data_extractor.py
    2-train_visualizer.py
    3-PH_pipeline.py
    4-analysis_plot.py
    5-analysis_statistics.py
    Meta_Analysis/
      contrast-analysis.py
      feedback-analysis.py
      LR-analyis.py

Requirements
------------
Python 3.9+ recommended. Example install:
  pip install torch torchvision numpy scipy matplotlib scikit-learn giotto-tda pyinform pandas one-api

Notes:
- giotto-tda provides Flagser for directed PH.
- IBL scripts use the International Brain Laboratory ONE API.

Usage
---------------
All scripts expose configuration via --help. Typical flow:
  # Logic gates – train LIF SNN and save activity
  python projects/logic_gates/0-and_or_xor_snn.py --help

  # MNIST – choose variant (0a/0b/0c) or create visuals (0d)
  python projects/mnist/0a-mnist_snn.py --help

  # IBL – extract data, compute TE→PH→AUBC, then analyze
  python projects/mouse_ibl/1-data_extractor.py --help
  python projects/mouse_ibl/3-PH_pipeline.py --help
  python projects/mouse_ibl/4-analysis_plot.py --help
  python projects/mouse_ibl/5-analysis_statistics.py --help
  python projects/mouse_ibl/Meta_Analysis/contrast-analysis.py --help
Outputs
-------
Most scripts write results inside an experiment root folder (passed via --root or created by the training scripts). Typical subfolders:

- learning/epoch_*/               per-epoch spike activity and cached TE/PH artifacts (when enabled)
- evaluation/test_*/              per-sample spike activity and cached TE/PH artifacts
- diagrams/                       figures and summary CSVs produced by the analysis scripts

The analysis scripts are designed to be rerunnable: they cache adjacency matrices and persistence diagrams per sample/epoch and will reuse them unless --overwrite_existing is set.
