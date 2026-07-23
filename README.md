# Predicting Water Pump Functionality in Tanzania with PySpark

Big Data & Machine Learning coursework project. Predict whether a rural water point is functional, non-functional or functional but needs repair, using PySpark's DataFrame API, ML Pipelines, RDDs and pytest test suite.

## Data
Source: Taarifa / Tanzanian Ministry of Water, distributed via DrivenData's "Pump it Up: Data Mining the Water Table" competition.
Not included in this repository (see '.gitignore') - place the 3 CSVs
('Training_set_values.csv', 'Training_set_labels.csv', 'Test_set_values.csv') in 'data/' before running

## Setup
\`\`\`bash
pip install -r requirements.txt
\`\`\`

## Run
\`\`\`bash
python -m src.train
\`\`\`

## Test
\`\`\`bash
pytest tests/
\`\`\`