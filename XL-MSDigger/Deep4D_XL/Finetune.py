from Deep4D_XL.dataset.Crosslink_Encoding_ccs import crosslink_ccs_encoding_with_label as encode_ccs
from Deep4D_XL.dataset.Crosslink_Encoding_msms import encoding_with_label as encode_msms
from Deep4D_XL.dataset.Crosslink_Encoding_rt import crosslink_rt_encoding_with_label as encode_rt
from Deep4D_XL.train_msms import do_train as train_msms
from Deep4D_XL.train_ccs import do_train as train_ccs
from Deep4D_XL.train_rt import do_train as train_rt
import os

class train_model():
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.load_msms_param_dir = os.path.join(base_dir, 'checkpoint', 'msms.pth')
        self.load_ccs_param_dir = os.path.join(base_dir, 'checkpoint', 'ccs.pth')
        self.load_rt_param_dir = os.path.join(base_dir, 'checkpoint', 'rt.pth')
        self.epochs = 30
        self.batch_size = 200
        self.ccs_lr = 0.00001
        self.rt_lr = 0.00001
        self.msms_lr = 0.0001
        self.vali_rate = 0.1

    def feature_encoding(self, msms_filedir, ccs_filedir, rt_filedir):
        feature_file = encode_msms(msms_filedir)
        encode_rt(rt_filedir)
        encode_ccs(ccs_filedir)
        return feature_file

    def train(self, feature_file):
        batch_size_adj = self.batch_size

        supervised_counts = {}

        for feature_name in ('rt', 'ccs'):
            feature_dir = os.path.join(feature_file, feature_name)

            if not os.path.isdir(feature_dir):
                raise RuntimeError(
                    f"Missing encoded {feature_name} directory: {feature_dir}"
                )

            supervised_counts[feature_name] = len([
                f
                for f in os.listdir(feature_dir)
                if os.path.isfile(os.path.join(feature_dir, f))
            ])

        min_supervised_files = min(supervised_counts.values())
        adjusted_size = int(min_supervised_files * self.vali_rate)

        if adjusted_size <= 0:
            raise RuntimeError(
                "CCS fine-tuning has too few encoded supervised examples "
                f"for validation: {supervised_counts}"
            )

        if adjusted_size < self.batch_size:
            batch_size_adj = adjusted_size
            print(
                f"[INFO] Batch size adjusted to {batch_size_adj} "
                f"based on CCS/RT encoded counts {supervised_counts}."
            )

        msms_paradir = train_msms(feature_file, self.load_msms_param_dir, self.epochs, batch_size_adj, self.msms_lr, self.vali_rate)
        ccs_paradir = train_ccs(feature_file, self.load_ccs_param_dir, self.epochs, batch_size_adj, self.ccs_lr, self.vali_rate)
        rt_paradir = train_rt(feature_file, self.load_rt_param_dir, self.epochs, batch_size_adj, self.rt_lr, self.vali_rate)
        return msms_paradir, ccs_paradir, rt_paradir

    def finetune(self, msms_filedir, ccs_filedir, rt_filedir):
        feature_file = self.feature_encoding(msms_filedir, ccs_filedir, rt_filedir)
        msms_paradir, ccs_paradir, rt_paradir = self.train(feature_file)
        return msms_paradir, ccs_paradir, rt_paradir

if __name__ == '__main__':
    x =train_model()
    x.feature_encoding('H:/20230708_new/test/305_plink_rt.csv',
                       'H:/20230708_new/test/305_plink_ccs.csv',
                       'H:/20230708_new/test/305_plink_complete_normal_library.csv', )
