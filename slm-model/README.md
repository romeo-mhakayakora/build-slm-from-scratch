## Model Artifacts

Large archives are split into parts to stay under GitHub's 100 MB file limit.

### Files in this folder

- `best_model_params.zip.part001`
- `best_model_params.zip.part002`
- `train.zip.part001`
- `train.zip.part002`
- `train.zip.part003`
- `train.zip.part004`
- `train.zip.part005`
- `validation.zip`

### Rebuild the full zip files (PowerShell)

```powershell
Get-Content .\best_model_params.zip.part* -Encoding Byte -ReadCount 0 | Set-Content .\best_model_params.zip -Encoding Byte
Get-Content .\train.zip.part* -Encoding Byte -ReadCount 0 | Set-Content .\train.zip -Encoding Byte
```

Then extract:

```powershell
Expand-Archive .\best_model_params.zip -DestinationPath .\best_model_params -Force
Expand-Archive .\train.zip -DestinationPath .\train -Force
Expand-Archive .\validation.zip -DestinationPath .\validation -Force
```
