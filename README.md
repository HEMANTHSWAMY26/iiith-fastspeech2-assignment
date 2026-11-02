# FastSpeech 2 Text-to-Speech Synthesis (English & Telugu)

Submission for: Intern / Research Assistant Evaluation — Speech Processing Lab (SLFI), LTRC, IIIT Hyderabad
Datasets: AI4Bharat Svarah (English, 5.0h Train / 1.0h Val) & IITM IndicTTS (Telugu, 5.0h Train / 1.0h Val)

## 1. Repository Folder Structure
- English/ : train.py, preprocess.py, inference.py, config.yaml, checkpoints/
- Telugu/  : train.py, preprocess.py, inference.py, config.yaml, checkpoints/
- Results/English/ : training_loss.png, validation_loss.png, test_generated_audios/, variance_adaptor_plots/
- Results/Telugu/  : training_loss.png, validation_loss.png, test_generated_audios/, variance_adaptor_plots/
- Report/Technical_Report.pdf
- requirements.txt

## 2. Reproducibility Commands
1. Install dependencies:
   pip install -r requirements.txt
2. Run preprocessing (extracts Mel, F0 Pitch, Energy, and Durations):
   python English/preprocess.py English/config.yaml
   python Telugu/preprocess.py Telugu/config.yaml
3. Train models:
   python English/train.py English/config.yaml
   python Telugu/train.py Telugu/config.yaml
4. Run inference on test sentences (E1-E4 & T1-T4) and generate Variance Adaptor plots:
   python English/inference.py English/config.yaml
   python Telugu/inference.py Telugu/config.yaml

## 3. Quantitative Training Summary
- English (Svarah): 3,189 train (5.0h) / 617 val (1.0h) utterances | Vocab=78 | Best Val Loss: 3.8367
- Telugu (IndicTTS): 2,530 train (5.0h) / 424 val (1.0h) utterances | Vocab=71 | Best Val Loss: 3.7000

Note on Model Checkpoints: Trained checkpoints (fastspeech2_best.pt) are stored inside English/checkpoints/ and Telugu/checkpoints/.