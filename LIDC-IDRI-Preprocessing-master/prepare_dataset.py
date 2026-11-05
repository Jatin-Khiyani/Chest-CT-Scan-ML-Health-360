import sys
import os
from pathlib import Path
import glob
from configparser import ConfigParser
import argparse
import pandas as pd
import numpy as np
import warnings
from statistics import median_high

# Ensure current script directory is in sys.path
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from utils import is_dir_path, segment_lung

# Locate configuration file (check cwd, script directory, workspace parent)
def find_conf_file(custom_path=None):
    if custom_path and os.path.exists(custom_path):
        return os.path.abspath(custom_path)
    candidates = [
        Path.cwd() / 'lung.conf',
        script_dir / 'lung.conf',
        script_dir.parent / 'lung.conf'
    ]
    for c in candidates:
        if c.exists():
            return str(c.resolve())
    raise FileNotFoundError("Could not find lung.conf. Please run config_file_create.py first.")

conf_file = find_conf_file()
parser = ConfigParser()
parser.read(conf_file)

# Ensure pylidc knows where config/dicom is
os.environ['PYLIDC_CONF'] = conf_file

import pylidc as pl
from pylidc.utils import consensus
from tqdm import tqdm

warnings.filterwarnings(action='ignore')

def ensure_dir(path_str):
    p = Path(path_str).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return str(p)

# Get Directory settings
DICOM_DIR = is_dir_path(parser.get('prepare_dataset', 'LIDC_DICOM_PATH'))
MASK_DIR = ensure_dir(parser.get('prepare_dataset', 'MASK_PATH'))
IMAGE_DIR = ensure_dir(parser.get('prepare_dataset', 'IMAGE_PATH'))
CLEAN_DIR_IMAGE = ensure_dir(parser.get('prepare_dataset', 'CLEAN_PATH_IMAGE'))
CLEAN_DIR_MASK = ensure_dir(parser.get('prepare_dataset', 'CLEAN_PATH_MASK'))
META_DIR = ensure_dir(parser.get('prepare_dataset', 'META_PATH'))

# Hyper Parameter settings
mask_threshold = parser.getint('prepare_dataset', 'Mask_Threshold')
confidence_level = parser.getfloat('pylidc', 'confidence_level')
padding = parser.getint('pylidc', 'padding_size')

class MakeDataSet:
    def __init__(self, LIDC_Patients_list, IMAGE_DIR, MASK_DIR, CLEAN_DIR_IMAGE, CLEAN_DIR_MASK, META_DIR, mask_threshold, padding, confidence_level=0.5, resume=False):
        self.IDRI_list = LIDC_Patients_list
        self.img_path = IMAGE_DIR
        self.mask_path = MASK_DIR
        self.clean_path_img = CLEAN_DIR_IMAGE
        self.clean_path_mask = CLEAN_DIR_MASK
        self.meta_path = META_DIR
        self.mask_threshold = mask_threshold
        self.c_level = confidence_level
        self.padding = [(padding, padding), (padding, padding), (0, 0)]
        self.resume = resume
        self.meta_cols = ['patient_id', 'nodule_no', 'slice_no', 'original_image', 'mask_image', 'malignancy', 'is_cancer', 'is_clean']
        self.meta_records = []
        self.meta_csv_path = os.path.join(self.meta_path, 'meta_info.csv')

        # Load existing meta records if resuming
        if self.resume and os.path.exists(self.meta_csv_path):
            try:
                existing_df = pd.read_csv(self.meta_csv_path, dtype={'patient_id': str, 'slice_no': str})
                # Re-zero-pad patient_id and slice_no just in case
                existing_df['patient_id'] = existing_df['patient_id'].astype(str).str.zfill(4)
                existing_df['slice_no'] = existing_df['slice_no'].astype(str).str.zfill(3)
                self.meta_records = existing_df.to_dict('records')
                print(f"Loaded {len(self.meta_records)} existing meta records for resume.")
            except Exception as e:
                print(f"Warning: Could not read existing meta_info.csv: {e}")

    def calculate_malignancy(self, nodule):
        list_of_malignancy = [annotation.malignancy for annotation in nodule]
        malignancy = median_high(list_of_malignancy)
        if malignancy > 3:
            return malignancy, True
        elif malignancy < 3:
            return malignancy, False
        else:
            return malignancy, 'Ambiguous'

    def save_meta(self, meta_list):
        row_dict = dict(zip(self.meta_cols, meta_list))
        self.meta_records.append(row_dict)

    def flush_meta_to_csv(self):
        df = pd.DataFrame(self.meta_records, columns=self.meta_cols)
        df.drop_duplicates(subset=['patient_id', 'nodule_no', 'slice_no'], keep='last', inplace=True)
        df.to_csv(self.meta_csv_path, index=False)

    def prepare_dataset(self):
        prefix = [str(x).zfill(3) for x in range(1000)]

        IMAGE_DIR = Path(self.img_path)
        MASK_DIR = Path(self.mask_path)
        CLEAN_DIR_IMAGE = Path(self.clean_path_img)
        CLEAN_DIR_MASK = Path(self.clean_path_mask)

        for patient in tqdm(self.IDRI_list, desc="Processing Patients"):
            pid = patient
            patient_image_dir = IMAGE_DIR / pid
            patient_mask_dir = MASK_DIR / pid
            patient_clean_dir_image = CLEAN_DIR_IMAGE / pid
            patient_clean_dir_mask = CLEAN_DIR_MASK / pid

            # If resuming and patient folders already have saved files, skip
            if self.resume:
                has_nodule_data = patient_image_dir.exists() and any(patient_image_dir.glob("*.npy"))
                has_clean_data = patient_clean_dir_image.exists() and any(patient_clean_dir_image.glob("*.npy"))
                if has_nodule_data or has_clean_data:
                    continue

            scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
            if scan is None:
                print(f"Scan for {pid} not found in database, skipping.")
                continue

            try:
                nodules_annotation = scan.cluster_annotations()
                vol = scan.to_volume()
            except Exception as e:
                print(f"Error loading DICOM for {pid}: {e}, skipping.")
                continue

            print("Patient ID: {} Dicom Shape: {} Number of Annotated Nodules: {}".format(pid, vol.shape, len(nodules_annotation)))

            Path(patient_image_dir).mkdir(parents=True, exist_ok=True)
            Path(patient_mask_dir).mkdir(parents=True, exist_ok=True)

            if len(nodules_annotation) > 0:
                for nodule_idx, nodule in enumerate(nodules_annotation):
                    mask, cbbox, masks = consensus(nodule, self.c_level, self.padding)
                    lung_np_array = vol[cbbox]
                    malignancy, cancer_label = self.calculate_malignancy(nodule)

                    for nodule_slice in range(mask.shape[2]):
                        if np.sum(mask[:, :, nodule_slice]) <= self.mask_threshold:
                            continue
                        lung_segmented_np_array = segment_lung(lung_np_array[:, :, nodule_slice])
                        lung_segmented_np_array[lung_segmented_np_array == -0] = 0

                        nodule_name = "{}_NI{}_slice{}".format(pid[-4:], prefix[nodule_idx], prefix[nodule_slice])
                        mask_name = "{}_MA{}_slice{}".format(pid[-4:], prefix[nodule_idx], prefix[nodule_slice])
                        meta_list = [pid[-4:], nodule_idx, prefix[nodule_slice], nodule_name, mask_name, malignancy, cancer_label, False]

                        self.save_meta(meta_list)
                        np.save(patient_image_dir / nodule_name, lung_segmented_np_array)
                        np.save(patient_mask_dir / mask_name, mask[:, :, nodule_slice])
            else:
                print("Clean Dataset", pid)
                Path(patient_clean_dir_image).mkdir(parents=True, exist_ok=True)
                Path(patient_clean_dir_mask).mkdir(parents=True, exist_ok=True)

                for slice_idx in range(vol.shape[2]):
                    if slice_idx > 50:
                        break
                    lung_segmented_np_array = segment_lung(vol[:, :, slice_idx])
                    lung_segmented_np_array[lung_segmented_np_array == -0] = 0
                    lung_mask = np.zeros_like(lung_segmented_np_array)

                    nodule_name = "{}_CN001_slice{}".format(pid[-4:], prefix[slice_idx])
                    mask_name = "{}_CM001_slice{}".format(pid[-4:], prefix[slice_idx])
                    meta_list = [pid[-4:], slice_idx, prefix[slice_idx], nodule_name, mask_name, 0, False, True]

                    self.save_meta(meta_list)
                    np.save(patient_clean_dir_image / nodule_name, lung_segmented_np_array)
                    np.save(patient_clean_dir_mask / mask_name, lung_mask)

            # Periodically flush metadata so progress is never lost
            self.flush_meta_to_csv()

        self.flush_meta_to_csv()
        print(f"Saved metadata to {self.meta_csv_path}")

if __name__ == '__main__':
    parser_arg = argparse.ArgumentParser(description="Preprocess LIDC-IDRI dataset with pylidc")
    parser_arg.add_argument('--patient', type=str, default=None, help="Process a single patient (e.g. LIDC-IDRI-0001 or 0001)")
    parser_arg.add_argument('--limit', type=int, default=None, help="Number of patients to process")
    parser_arg.add_argument('--start', type=int, default=0, help="Start index in patient list")
    parser_arg.add_argument('--resume', action='store_true', help="Skip already processed patients")
    args = parser_arg.parse_args()

    valid_ids = set(scan.patient_id for scan in pl.query(pl.Scan).all())
    LIDC_IDRI_list = [f for f in os.listdir(DICOM_DIR) if f in valid_ids and os.path.isdir(os.path.join(DICOM_DIR, f))]
    LIDC_IDRI_list.sort()

    if args.patient:
        target = args.patient if args.patient.startswith("LIDC-IDRI-") else f"LIDC-IDRI-{args.patient.zfill(4)}"
        LIDC_IDRI_list = [p for p in LIDC_IDRI_list if p == target]
        if not LIDC_IDRI_list:
            print(f"Patient {target} not found in {DICOM_DIR}!")
            sys.exit(1)
    else:
        if args.start > 0:
            LIDC_IDRI_list = LIDC_IDRI_list[args.start:]
        if args.limit is not None:
            LIDC_IDRI_list = LIDC_IDRI_list[:args.limit]

    print(f"Total patients to process in this run: {len(LIDC_IDRI_list)}")

    test = MakeDataSet(
        LIDC_IDRI_list,
        IMAGE_DIR,
        MASK_DIR,
        CLEAN_DIR_IMAGE,
        CLEAN_DIR_MASK,
        META_DIR,
        mask_threshold,
        padding,
        confidence_level,
        resume=args.resume
    )
    test.prepare_dataset()
