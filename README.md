# TextDDI

Experiment code and data for: "Learning to Describe for Predicting Zero-shot Drug-Drug Interactions"

## Running Scripts

To work with the DrugBank dataset and train models, follow these instructions:

### Training DDI-Predictor
- Modify the configuration file at `configs/main_drugbank.yaml`.
- Run the script located at `drugbank/main_drugbank.py` by executing the following command in your terminal:
  ```
  python drugbank/main_drugbank.py
  ```

### Training Information-Selector

- Adjust the configuration in `configs/ppo_drugbank.yaml`. Ensure you update the `initialize_checkpoint` and `policy_checkpoint` to the trained DDI-Predictor weights.
- Execute the training script `drugbank/train_ppo_drugbank.py` by running:
  ```
  python drugbank/train_ppo_drugbank.py
  ```

Ensure all paths and dependencies are correctly set up before running the scripts.
