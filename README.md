# XL-MSDigger_fixed

> **Independent bug-fixed derivative of XL-MSDigger**
>
> This repository is an independently maintained modification of the original
> [Chen-micslab/XL-MSDigger](https://github.com/Chen-micslab/XL-MSDigger).
> It is not the official XL-MSDigger repository.
>
> The modifications in this repository address reproducibility and execution
> issues identified during validation of the DDA pLink2 workflow, including
> MGF preprocessing, candidate generation, DNN rescoring output preservation,
> and deterministic Deep4D-XL fine-tuning.
>
> Original XL-MSDigger authors and software remain credited under the
> repository's MIT License.

---

## Validated fixes and reproducibility

This repository contains targeted fixes to the DDA pLink2 workflow of XL-MSDigger. The corrected workflow was validated using the original XL-MSDigger pLink2 test dataset and an end-to-end run from pLink preprocessing through Deep4D-XL fine-tuning, candidate feature prediction, DNN rescoring, and final 1% FDR filtering.

The main validated fixes are:

- corrected package import handling in the pLink/MGF preprocessing module;
- single-pass MGF metadata indexing to avoid repeated full-file searches;
- support for retention time and ion-mobility metadata embedded in MGF `TITLE` fields;
- corrected and indexed MGF fragment-spectrum matching;
- generation of the all-candidate precursor and MS/MS files required for rescoring;
- restriction of Deep4D-XL fine-tuning to intra-protein cross-linked PSMs, while retaining all candidates for downstream rescoring;
- corrected six-value preprocessing return handling in the DDA driver;
- explicit protection against silently applying the no-CCS DDA workflow to data containing explicit ion-mobility metadata;
- preservation of the full DNN parameter-search table;
- preservation of preprocessing and model intermediates by default, with optional cleanup;
- deterministic Deep4D-XL MS/MS and RT fine-tuning on the validated CUDA/PyTorch/hardware stack.

### Deterministic training validation

Two independent Deep4D-XL fine-tuning runs were performed using identical intra-protein training inputs in separate fresh Python processes.

Both MS/MS checkpoints were byte-identical and had SHA256:

```text
9b08fa32ff39852c9f8e21ad69a623672c183156bdefd32104621954c975464a
```

Both RT checkpoints were byte-identical and had SHA256:

```text
f7be1713f249c65c5326c8d0df9b4a2d0a687915b0beadfe82c3ff0be60805c1
```

The same checkpoint hashes were reproduced during the subsequent complete end-to-end XL-MSDigger run.

### End-to-end DDA pLink2 validation

The final deterministic end-to-end run processed 76,077 candidate CSMs and completed successfully.

At 1% FDR for inter-protein target CSMs:

| Metric | Original pLink result | Fixed XL-MSDigger result |
|---|---:|---:|
| Target CSMs | 75 | 83 |
| Retained original targets | — | 65 |
| Newly identified targets | — | 18 |
| Lost original targets | — | 10 |
| Sensitivity to original targets | — | 86.7% |
| Net change in target CSM count | — | +10.7% |

The selected DNN rescoring configuration for this validation run used 2-fold cross-validation, 30 epochs, a learning rate of `0.01`, and a maximum training sample size of `2000`.

These values should be interpreted as validation results for the supplied test dataset rather than as expected identification counts for other datasets.

### Reproducibility scope

Deterministic checkpoint reproduction was validated on the same software and GPU stack used during testing, including PyTorch `2.0.1+cu118` on an NVIDIA H200 NVL GPU. PyTorch does not guarantee bitwise reproducibility across different releases, CUDA versions, platforms, or hardware, so identical checkpoint hashes should not be assumed across arbitrary environments.

The upstream XL-MSDigger project remains available at [Chen-micslab/XL-MSDigger](https://github.com/Chen-micslab/XL-MSDigger). This repository is an independently maintained bug-fixed derivative and is not an official release of the original project.

---

## Original XL-MSDigger documentation
Here, we constructed Deep4D-XL, a deep learning tool capable of accurately predicting cross-linked peptide’s multi-dimensional information, including retention time, collisional cross-section, fragment ion intensity. Using Deep4D-XL as the core, we developed XL-MSDigger, a pipeline for comprehensive analysis of cross-linking mass spectrometry data acquired through both DDA and DIA approaches.
## Environment Setup
Create a new conda environment first:
```
conda create --name XLMSDigger python=3.9
```
Activate this environment by running:
```
conda activate XLMSDigger
```
then install dependencies:
```
pip install -r ./requirements.txt
```
Please download the checkpoint folder from [checkpoint.zip](https://drive.google.com/file/d/1tXZIgpxKFgSOx_0K42nrSTh9U11vFkc7/view?usp=sharing) and extract it to the 'XL-MSDigger-main/Deep4D_XL/checkpoint' path.
## Run Instructions of XL-MSDigger
### DDA XL-MS analysis 
#### Rescoring of DDA XL-MS results
This step performs rescoring on the results generated by the DDA software. Now XL-MSDigger support plink2, pLink3 and Scout.
```
python XL-MSDigger_DDA_plink.py --plinkfile './test_data/plink_test' --mgf_dir './test_data/test.mgf' --rescore_model 'dnn'
```
Description of argparse:  
--plinkfile: The file directory of pLink output.  
--mgf_dir: The file directory of mgf file.  
--rescore_model: You can select 'dnn' and 'svm'.  
It can also export mzIdentML (v1.2) after rescoring. Provide `--fasta` and the script will write a `.mzid` next to the rescored CSV.
Test MGF data: [test.mgf](https://drive.google.com/file/d/1NkTArho0gmGIBLsbBz7VM5W4V1sap5WW/view?usp=sharing),
Test pLink2 results folder: [plink_test.zip](https://drive.google.com/file/d/1RhtQTLLn5a7OrKfJTYmioHocx_fxtLlq/view?usp=sharing).
### DIA XL-MS analysis 
#### Building spectral library  
This step is used to build a predicted spectral library for proteins (intra-protein crosslinks) or PPIs (inter-protein crosslinks of interest.
```
python Build_library.py --experiment_library './test_data/experimental_library.csv' --aim_protein './test_data/test_PPI.csv' --aim_type 1 --fasta_dir './test_data/human_reviewed.fasta'
```
Description of argparse:  
--experiment_library: The file directory of experimental library.  
--aim_protein: The file directory of aim protein or PPI.  
--aim_type: Type of target (protein or PPI). Use 0 for proteins or 1 for PPIs.  
--fasta_dir: The file directory of fasta file.   
Experimental_library data: [experimental_library](https://drive.google.com/file/d/1DmDQv-QUX7tvyAgu30jqn3srfQMghET_/view?usp=sharing),
test_PPI data: [test_PPI](https://drive.google.com/file/d/1pHGtpFXhumuXi_tg1BFoqvKly040mii6/view?usp=sharing),
fasta data: [fasta file](https://drive.google.com/file/d/1QC0zpgYdvGW22NvOXUvu5mulwFHyl_Tw/view?usp=sharing).
#### Rescoring of DIA XL-MS results
This step is used to rescore the output results from DIA-NN and output the identified cross-linked peptides belonging to proteins or protein-protein interactions of interest.
```
python XL-MSDigger_DIA.py --diann_report './test_data/report.tsv' --DIA_library './test_data/experimental_library_with_aim_normal_lib.csv' --fasta_dir './test_data/human_reviewed.fasta'  --peptide_protein_list './test_data/test_peptide&protein.csv'
```
Description of argparse:  
--diann_report: The file directory of DIA-NN report.  
--DIA_library: The spectral library file containing "with_aim_normal_lib.csv" in its name, generated by the "Building spectral library" step.  
--fasta_dir: The file directory of fasta file.  
--peptide_protein_list: The file containing "peptide&protein.csv" in its name, generated by the "Building spectral library" step.  
test diann_report: [report.tsv](https://drive.google.com/file/d/1vLsqdUGnaqKnAyt7aRDxJ0I2FBIj4GQu/view?usp=sharing),
DIA_library data: [DIA_library](https://drive.google.com/file/d/15IOg0bPPW7zpau6-PUt5N_wXoITI3tK8/view?usp=sharing),
fasta data: [fasta file](https://drive.google.com/file/d/1QC0zpgYdvGW22NvOXUvu5mulwFHyl_Tw/view?usp=sharing),
peptide_protein_list: [peptide&protein_list](https://drive.google.com/file/d/1Bkw8mOUupGk5F3ANTLWPIs2NZkjgA1e3/view?usp=sharing).

## Run Instructions of Deep4D-XL
This step is intended for users who want to independently train the MS/MS, RT, or CCS prediction models. Although XL-MSDigger already includes the ability to fine-tune models based on each file’s results, you can independently train these modules if you are working with a new cross-linker or have a large amount of batch-specific data.
### a. Train ccs model  
#### 1. Encoding 
Run `'Deep4D_XL/CCS/dataset/Crosslink_Encoding.py'`. 
```
python Crosslink_Encoding.py --filename 'CCS_train'
```
Note: CCS_train.csv must be placed under Deep4D_XL/CCS/dataset/data/
#### 2. Train ccs model
Run `'Deep4D_XL/CCS/train_ccs.py'` 
```
python train_ccs.py --filename 'CCS_train' --load_ccs_param_dir 'Deep4D_XL/CCS/checkpoint/ccs.pth' 
```
--filename: Training data name.  
--load_ccs_param_dir: The file directory of pre-trained ccs model.  
Finally, find the parameters file at 'Deep4D_XL/CCS/checkpoint/{filename}_ccs/', where you can select the model checkpoint with the lowest MedianRE.
### b. Train rt model  
#### 1. Encoding 
Run `'Deep4D_XL/RT/dataset/Crosslink_Encoding.py'`. 
```
python Crosslink_Encoding.py --filename 'RT_train'
```
Note: RT_train.csv must be placed under Deep4D_XL/RT/dataset/data/
#### 2. Train rt model
Run `'Deep4D_XL/RT/train_rt.py'` 
```
python train_rt.py --filename 'RT_train' --load_rt_param_dir 'Deep4D_XL/RT/checkpoint/rt.pth' 
```
--filename: Training data name.  
--load_rt_param_dir: The file directory of pre-trained rt model  
Finally, find the parameters file at 'Deep4D_XL/RT/checkpoint/{filename}_rt/', where you can select the model checkpoint with the lowest MAE.
### c. Train msms model  
#### 1. Encoding 
For the non-cleavable crosslinker, run `'RT/dataset/Crosslink_Encoding_NC.py'`. For the cleavable crosslinker, Run `'Deep4D_XL/MSMS/Crosslink_Encoding_C.py'`.
```
python Crosslink_Encoding_NC.py --filename 'MSMS_train'
```
Note: MSMS_train.csv must be placed under Deep4D_XL/MSMS/dataset/data
#### 2. Train msms model
For the non-cleavable crosslinker, run `'Deep4D_XL/MSMS/train_crosslink_msms_NC.py'`. For the cleavable crosslinker, Run `'Deep4D_XL/MSMS/train_crosslink_msms_C.py'`.
```
python train_crosslink_msms_NC.py --filename 'MSMS_train' --load_msms_param_dir 'Deep4D_XL/RT/checkpoint/msms_nc.pth' 
```
--filename: Training data name.  
--load_msms_param_dir: The file directory of pre-trained msms model  
Finally, find the parameters file at 'Deep4D_XL/MSMS/checkpoint/{filename}_msms/', where you can select the model checkpoint with the highest dot product.
## Contacts
Please report any problems directly to the github issue tracker. Also, you can send feedback to moran.chen@bcm.edu.
