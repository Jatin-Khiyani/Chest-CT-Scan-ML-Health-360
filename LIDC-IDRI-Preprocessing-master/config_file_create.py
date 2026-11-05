import os
from pathlib import Path
from configparser import ConfigParser

def create_config():
    # Resolve project root and paths
    script_dir = Path(__file__).resolve().parent
    workspace_dir = script_dir.parent

    # Check for lidc_idri folder
    if (workspace_dir / "lidc_idri").exists():
        dicom_dir = (workspace_dir / "lidc_idri").resolve()
    elif (script_dir / "lidc_idri").exists():
        dicom_dir = (script_dir / "lidc_idri").resolve()
    elif (workspace_dir / "LIDC-IDRI").exists():
        dicom_dir = (workspace_dir / "LIDC-IDRI").resolve()
    else:
        dicom_dir = Path("E:/LIRDC/lidc_idri")

    data_dir = script_dir / "data"

    config = ConfigParser()

    # prepare_dataset.py configuration
    config['prepare_dataset'] = {
        # Path To LIDC Dataset
        'LIDC_DICOM_PATH': str(dicom_dir).replace('\\', '/'),
        # Directory to save output files
        'MASK_PATH': str(data_dir / 'Mask').replace('\\', '/'),
        'IMAGE_PATH': str(data_dir / 'Image').replace('\\', '/'),
        'CLEAN_PATH_IMAGE': str(data_dir / 'Clean' / 'Image').replace('\\', '/'),
        'CLEAN_PATH_MASK': str(data_dir / 'Clean' / 'Mask').replace('\\', '/'),
        'META_PATH': str(data_dir / 'Meta').replace('\\', '/'),
        'Mask_Threshold': '8'
    }

    # Configuration for pylidc library
    config['pylidc'] = {
        'confidence_level': '0.5',
        'padding_size': '512'
    }

    # pylidc dicom section
    config['dicom'] = {
        'path': str(dicom_dir).replace('\\', '/')
    }

    # Ensure output directories exist
    for p in ['Mask', 'Image', 'Clean/Image', 'Clean/Mask', 'Meta']:
        (data_dir / p).mkdir(parents=True, exist_ok=True)

    # Write lung.conf to script directory and workspace root
    conf_paths = [script_dir / "lung.conf", workspace_dir / "lung.conf"]
    for conf_path in conf_paths:
        with open(conf_path, 'w') as f:
            config.write(f)
        print(f"Configuration written to: {conf_path}")

    # Also update pylidc.conf
    pylidc_paths = [script_dir / "pylidc.conf", workspace_dir / "pylidc.conf"]
    pylidc_config = ConfigParser()
    pylidc_config['dicom'] = {'path': str(dicom_dir).replace('\\', '/')}
    for p_path in pylidc_paths:
        with open(p_path, 'w') as f:
            pylidc_config.write(f)
        print(f"pylidc configuration written to: {p_path}")

if __name__ == "__main__":
    create_config()
