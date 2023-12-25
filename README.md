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

- Adjust the configuration in `configs/ppo_drugbank.yaml`. Ensure you update the `initialize_checkpoint` and `policy_checkpoint` to the trained DDI-Predictor checkpoint path.
- Execute the training script `drugbank/train_ppo_drugbank.py` by running:
  ```
  python drugbank/train_ppo_drugbank.py
  ```

Ensure all paths and dependencies are correctly set up before running the scripts.

## Reference
```
@inproceedings{zhu-etal-2023-learning,
    title = "Learning to Describe for Predicting Zero-shot Drug-Drug Interactions",
    author = "Zhu, Fangqi and Zhang, Yongqi and Chen, Lei and Qin, Bing and Xu, Ruifeng",
    editor = "Bouamor, Houda and Pino, Juan and Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.918",
    doi = "10.18653/v1/2023.emnlp-main.918",
    pages = "14855--14870",
}
```
