## SLM Model Usage Notes

This folder contains the trained parameter checkpoint and a notebook used for local inference experiments.

### Files

- `best_model_params.pt`: trained model checkpoint.
- `best_model_params.zip.part001`
- `best_model_params.zip.part002`
- `best_model_params.zip`: reconstructed archive form of the checkpoint.
- `Vizuara_AI_Labs_Small_Language_Model_Scratch_Final_(2) (1).ipynb`: notebook used for testing generation.

### Rebuild Checkpoint Zip (if needed)

If you only have the split parts, rebuild the archive first:

```powershell
Get-Content .\best_model_params.zip.part* -Encoding Byte -ReadCount 0 | Set-Content .\best_model_params.zip -Encoding Byte
Expand-Archive .\best_model_params.zip -DestinationPath .\best_model_params -Force
```

### Load + Quick Inference

When checkpoint loading works, you should see:

```text
<All keys matched successfully>
```

Then test with prompts such as:

- `romeo mhakayakora.`
- `A little girl went to the woods`

### What To Expect From Outputs

Current generations are coherent in short spans but can drift over longer text (topic jumps, repetition, mixed grammar). This is expected for a small model trained from scratch with limited data/compute.

### Environment Note

If you run locally (not Google Colab), this import will fail:

```python
from google.colab import runtime
```

Error:

```text
ModuleNotFoundError: No module named 'google.colab'
```

Use standard local notebook/kernel controls instead of Colab runtime commands.
