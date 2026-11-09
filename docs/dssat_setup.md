# DSSAT setup on Windows

1. Install DSSAT v4.8.6 using the official installer. Keep the executable and example data together, commonly under `C:\DSSAT48`.
2. Confirm `DSCSM048.EXE`, `DSSATPRO.v48`, the crop folders you need, `Weather`, `Soil`, and `Genotype` exist under the installation.
3. Configure the project. `.env` at the project root is loaded automatically; process environment variables override it. Relative configured project-data/model paths are based on the project root, not the shell's current directory. Empty optional values are ignored. For example, edit the copied `.env` file:

   ```dotenv
   DSSAT_HOME=C:\DSSAT48
   DSSAT_DATA=C:\DSSAT48
   PROJECT_DATA_DIR=data
   MODEL_DIR=models
   ```

   Or set paths in PowerShell:

   ```powershell
   $env:DSSAT_HOME = 'C:\DSSAT48'
   # Or specify the executable and data root independently:
   $env:DSSAT_EXECUTABLE = 'C:\DSSAT48\DSCSM048.EXE'
   $env:DSSAT_DATA = 'C:\DSSAT48'
   ```

4. Run `python -m cli.main doctor`, then `python -m cli.main list-experiments --crop all`.
5. `python -m cli.main validate-experiment --experiment KSAS8101` checks only simulation inputs. Add `--for-dataset` to additionally require a parseable measured-yield `.WHA` file.
6. Run `python -m cli.main run-experiment --experiment KSAS8101 --treatment 1`. This simulation does not need observed yields.

The adapter invokes the DSSAT batch pattern with the module selected for the FileX crop (for example, the Wheat module is `CSCER048`). DSSAT's profile contains paths to shared soil, weather, and genotype directories. The app stages the selected FileX and profile in its isolated run folder, executes DSSAT with that folder as its working directory, and does not edit the installed source data. Outputs are archived under `data/runs/<run-id>/output/`; diagnostic stdout/stderr and metadata remain beside them. If the executable is not detected, set `DSSAT_HOME` to its parent directory or set `DSSAT_EXECUTABLE` to the executable itself.
