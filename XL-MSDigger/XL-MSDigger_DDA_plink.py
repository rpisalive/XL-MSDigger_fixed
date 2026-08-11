import argparse
from Preprocess.plink_with_msconvert_mgf import plink_with_msconvert_mgf
from Deep4D_XL.Finetune_noccs import train_model as train_model_noccs
from Deep4D_XL.Finetune import train_model as train_model_ccs
from DDA_rescore.predict_feature_noccs import generate_feature as generate_feature_noccs
from DDA_rescore.predict_feature import generate_feature as generate_feature_ccs
from DDA_rescore.DDA_rescore_plink3 import Rescore_SVM, Rescore_DNN
from DDA_rescore.mzid_writer import build_mzid
import os
import pandas as pd
import time 

DEFAULT_MOD_INI = "/Users/moranchen/Documents/Project/Deep4D_XL/Review_data/data/mzidentML/modification.ini"

def get_args():             
    parser = argparse.ArgumentParser(description='Train the transformer on peptide and ccs')
    parser.add_argument('--plinkfile', type=str, default='/data/plinkfile')
    parser.add_argument('--mgf_dir', type=str, default='/data/plinkfile.mgf')
    parser.add_argument('--finetune', type=int, default=1)
    parser.add_argument('--rescore_model', type=str, default='dnn')
    parser.add_argument('--rescore_fdr', type=float, default=0.01)
    parser.add_argument('--rescore_batch_size', type=int, default=200)
    parser.add_argument('--rescore_vali_rate', type=float, default=0.1)
    parser.add_argument('--rescore_model_parameter', type=str, default=None)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--fasta', type=str, default=None)
    parser.add_argument('--mzid', type=int, default=1)
    parser.add_argument(
        '--cleanup_intermediates',
        type=int,
        choices=[0, 1],
        default=0,
        help=(
            'Delete preprocessing, encoding, and checkpoint '
            'intermediates after a successful run. '
            'Default: 0 (preserve intermediates).'
        )
    )
    parser.add_argument('--mod_ini', type=str, default=DEFAULT_MOD_INI)
    return parser.parse_args()

def find_plink_params(plink_path):
    if os.path.isfile(plink_path) and plink_path.lower().endswith(".plink"):
        return plink_path
    if os.path.isdir(plink_path):
        candidates = sorted(
            os.path.join(plink_path, name)
            for name in os.listdir(plink_path)
            if name.lower().endswith(".plink")
        )
        return candidates[0] if candidates else None
    return None

def run():
    args = get_args()          
    plink = plink_with_msconvert_mgf()
    (
        msms_dir,
        ccs_dir,
        rt_dir,
        candidate_msms_dir,
        candidate_rtccs_dir,
        has_ion_mobility
    ) = plink.process(
        args.plinkfile,
        args.mgf_dir
    )

    # --------------------------------------------------
    # Select the Deep4D-XL DDA workflow from the
    # ion-mobility metadata detected during preprocessing.
    #
    # no-CCS:
    #   MS/MS + RT
    #
    # CCS-aware:
    #   MS/MS + RT + CCS
    # --------------------------------------------------

    base_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    if has_ion_mobility:

        print(
            "DDA prediction mode: CCS-aware "
            "(RT + CCS + MS/MS)"
        )

        if args.finetune == 1:

            print(
                "Finetuning MS/MS, CCS, and RT models......"
            )

            train = train_model_ccs()

            (
                msms_paradir,
                ccs_paradir,
                rt_paradir
            ) = train.finetune(
                msms_dir,
                ccs_dir,
                rt_dir
            )

        else:

            print(
                "No Finetuning: using bundled "
                "MS/MS, CCS, and RT checkpoints"
            )

            msms_paradir = os.path.join(
                base_dir,
                "Deep4D_XL",
                "checkpoint",
                "msms.pth"
            )

            ccs_paradir = os.path.join(
                base_dir,
                "Deep4D_XL",
                "checkpoint",
                "ccs.pth"
            )

            rt_paradir = os.path.join(
                base_dir,
                "Deep4D_XL",
                "checkpoint",
                "rt.pth"
            )

            required_model_files = [
                msms_paradir,
                ccs_paradir,
                rt_paradir
            ]

            missing_model_files = [
                model_file
                for model_file in required_model_files
                if not os.path.isfile(model_file)
            ]

            if missing_model_files:
                raise RuntimeError(
                    "CCS-aware DDA prediction requires "
                    "bundled MS/MS, CCS, and RT checkpoints. "
                    f"Missing: {missing_model_files}"
                )

        generate = generate_feature_ccs()

        candidate_feature = generate.run(
            candidate_msms_dir,
            candidate_rtccs_dir,
            msms_paradir,
            ccs_paradir,
            rt_paradir
        )

        if "ccs_RE" not in candidate_feature.columns:
            raise RuntimeError(
                "CCS-aware DDA feature generation completed "
                "without producing the required ccs_RE feature."
            )

        print(
            "CCS feature generated:",
            "ccs_RE"
        )

        print(
            "CCS feature missing values:",
            candidate_feature["ccs_RE"].isna().sum()
        )

    else:

        print(
            "DDA prediction mode: no-CCS "
            "(RT + MS/MS)"
        )

        if args.finetune == 1:

            print(
                "Finetuning MS/MS and RT models......"
            )

            train = train_model_noccs()

            (
                msms_paradir,
                rt_paradir
            ) = train.finetune(
                msms_dir,
                rt_dir
            )

        else:

            print("No Finetuning")

            # Preserve the previously validated
            # no-CCS checkpoint behaviour.
            msms_paradir = os.path.join(
                base_dir,
                "Deep4D_XL",
                "checkpoint",
                "PXD017620",
                "MSMS.pth"
            )

            rt_paradir = os.path.join(
                base_dir,
                "Deep4D_XL",
                "checkpoint",
                "PXD017620",
                "RT.pth"
            )

        generate = generate_feature_noccs()

        candidate_feature = generate.run(
            candidate_msms_dir,
            candidate_rtccs_dir,
            msms_paradir,
            rt_paradir
        )

    # --------------------------------------------------
    # pLink compatibility aliases required by the
    # DDA rescoring implementation.
    #
    # Validation against the original pLink output
    # established:
    #   Re-score_CSM = score
    #   Q-value_CSM  = Q-value
    # --------------------------------------------------

    required_source_columns = [
        "score",
        "Q-value"
    ]

    missing_source_columns = [
        col
        for col in required_source_columns
        if col not in candidate_feature.columns
    ]

    if missing_source_columns:
        raise RuntimeError(
            "Cannot prepare DDA rescoring features. "
            f"Missing columns: {missing_source_columns}"
        )

    candidate_feature["Re-score_CSM"] = (
        candidate_feature["score"]
    )

    candidate_feature["Q-value_CSM"] = (
        candidate_feature["Q-value"]
    )

    print(
        "Candidate feature rows:",
        len(candidate_feature)
    )

    print(
        "Re-score_CSM equals score:",
        candidate_feature["Re-score_CSM"]
        .equals(candidate_feature["score"])
    )

    print(
        "Q-value_CSM equals Q-value:",
        candidate_feature["Q-value_CSM"]
        .equals(candidate_feature["Q-value"])
    )

    if (
        "Target_Decoy" in candidate_feature.columns
    ):
        original_positive_count = (
            (
                candidate_feature["Target_Decoy"] == 2
            )
            &
            (
                candidate_feature["Q-value_CSM"]
                <= args.rescore_fdr
            )
        ).sum()

        print(
            "Original target positives at rescoring FDR:",
            original_positive_count
        )

    candidate_feature_dir = (
        candidate_rtccs_dir.split(".csv")[0]
        + "_candidate_feature.csv"
    )
    candidate_feature.to_csv(candidate_feature_dir, index=False)
    candidate_feature = pd.read_csv(candidate_feature_dir)
    if args.rescore_model == 'svm':
        print('SVM will be used for rescoring')
        dda_rescore = Rescore_SVM()
        rescore_results = dda_rescore.run(candidate_feature)
    elif args.rescore_model == 'dnn':
        print('DNN will be used for rescoring')
        dda_rescore= Rescore_DNN()
        rescore_results = dda_rescore.run(args, candidate_feature, candidate_rtccs_dir)
    else:
        print('Invalid model name entered, SVM will be used for rescoring')
        dda_rescore = Rescore_SVM()
        rescore_results = dda_rescore.run(candidate_feature)
    rescore_results_dir = candidate_rtccs_dir.split('.csv')[0] + '_rescore_results.csv'
    rescore_results.to_csv(rescore_results_dir, index=False)
    rescore_results = rescore_results[rescore_results['FDR'] <= args.rescore_fdr]
    rescore_results = rescore_results[rescore_results['Target_Decoy'] == 2]
    rescore_results_dir = candidate_rtccs_dir.split('.csv')[0] + '_rescore_results1.csv'
    rescore_results.to_csv(rescore_results_dir, index=False)
    if args.mzid == 1:
        if not args.fasta:
            print('WARN: --fasta not provided; skipping mzIdentML export.')
        else:
            plink_params_path = find_plink_params(args.plinkfile)
            if not plink_params_path:
                print('WARN: No .plink params file found; skipping mzIdentML export.')
            elif not os.path.isfile(args.mod_ini):
                print(f'WARN: modification.ini not found at {args.mod_ini}; skipping mzIdentML export.')
            else:
                mzid_out = rescore_results_dir.rsplit('.', 1)[0] + '.mzid'
                build_mzid(rescore_results_dir, args.fasta, args.mgf_dir, plink_params_path, args.mod_ini, mzid_out)
                print(f'mzIdentML saved: {mzid_out}')
    # --------------------------------------------------
    # Optional cleanup
    #
    # Preserve intermediates by default so preprocessing,
    # fine-tuning, prediction, and rescoring outputs can be
    # audited and reproduced.
    # --------------------------------------------------

    if args.cleanup_intermediates == 1:
        print(
            "Cleaning intermediate files and directories..."
        )

        intermediate_files = [
            msms_dir,
            ccs_dir,
            rt_dir,
            candidate_msms_dir,
            candidate_rtccs_dir
        ]

        for intermediate_file in intermediate_files:
            if os.path.isfile(intermediate_file):
                os.remove(intermediate_file)

        folder_path = os.path.dirname(
            os.path.abspath(args.mgf_dir)
        )

        import shutil

        intermediate_directories = [
            os.path.join(
                folder_path,
                "candidate_feature_encoding"
            ),
            os.path.join(
                folder_path,
                "checkpoint"
            ),
            os.path.join(
                folder_path,
                "feature_encoding"
            ),
        ]

        for intermediate_directory in intermediate_directories:
            shutil.rmtree(
                intermediate_directory,
                ignore_errors=True
            )

        print("Intermediate cleanup completed.")

    else:
        print(
            "Intermediate files preserved "
            "(--cleanup_intermediates 0)."
        )

if __name__ == '__main__':
    run()
