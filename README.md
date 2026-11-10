# Massive Activations Confound Sparse Autoencoders

Code and saved results for the paper.

## Start here

To check the saved results, run these commands from this folder:

```bash
python -m pip install -r requirements.txt
python analysis/check_results.py
```

## Files

| Folder | Contents |
|---|---|
| [experiments/](experiments/README.md) | Experiment scripts, listed by paper section. |
| [analysis/](analysis/README.md) | Scripts for checking results and making tables. |
| [results/](results/README.md) | Saved measurements and tables. |
| [pc1_sae/](pc1_sae/README.md) | Load an SAE and apply PC1 bypass. |
| [checkpoints/](checkpoints/README.md) | Checkpoint locations, revisions, and file hashes. |

## Running experiments

The experiment guide names the script and result files for each section. GPU runs need PyTorch and the packages in `requirements-models.txt`.

## License

Code: [MIT](LICENSE). Results and SAE weights: [CC BY 4.0](LICENSE-DATA.md). Base models retain their own licenses.
