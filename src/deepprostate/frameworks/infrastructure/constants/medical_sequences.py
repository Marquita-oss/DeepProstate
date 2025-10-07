from typing import Dict, Set

SEQUENCE_DESCRIPTIONS = {
    'T1W': 'T1-Weighted MRI',
    'T2W': 'T2-Weighted MRI',
    'FLAIR': 'FLAIR MRI',
    'DWI': 'Diffusion Weighted Imaging',
    'ADC': 'Apparent Diffusion Coefficient',
    'DCE': 'Dynamic Contrast Enhanced',
    'HBV': 'High B-Value DWI',
    'CT_ARTERIAL': 'CT Arterial Phase',
    'CT_VENOUS': 'CT Venous Phase',
    'CT_DELAYED': 'CT Delayed Phase',
    'CT_NATIVE': 'CT Native/Unenhanced'
}

SEQUENCE_MAPPINGS = {
    't2w': 'T2W', 't2': 'T2W', 't2_w': 'T2W',
    'adc': 'ADC', 'adc_map': 'ADC',
    'dwi': 'DWI', 'dw': 'DWI', 'diffusion': 'DWI',
    'hbv': 'HBV', 'high_b': 'HBV', 'highb': 'HBV',
    't1w': 'T1W', 't1': 'T1W', 't1_w': 'T1W',
    'flair': 'FLAIR',
    'dce': 'DCE', 'dynamic': 'DCE', 'contrast': 'DCE'
}

MRI_SEQUENCES = {'T1W', 'T2W', 'FLAIR', 'DWI', 'ADC', 'DCE', 'HBV'}

CT_SEQUENCES = {'CT_ARTERIAL', 'CT_VENOUS', 'CT_DELAYED', 'CT_NATIVE'}

PROSTATE_REQUIRED_SEQUENCES = {'T2W', 'ADC'}

PROSTATE_OPTIONAL_SEQUENCES = {'HBV', 'DWI', 'DCE'}

SEQUENCE_INDICATORS = {
    'T1W': '[T1]',
    'T2W': '[T2]', 
    'FLAIR': '[FL]',
    'DWI': '[DW]',
    'ADC': '[AD]',
    'DCE': '[DC]',
    'HBV': '[HB]',
    'CT_ARTERIAL': '[CA]',
    'CT_VENOUS': '[CV]',
    'CT_DELAYED': '[CD]',
    'CT_NATIVE': '[CN]'
}

def get_sequence_description(sequence_type: str, fallback: str = None) -> str:
    return SEQUENCE_DESCRIPTIONS.get(sequence_type, fallback or f'{sequence_type} Sequence')

def get_sequence_indicator(sequence_type: str) -> str:
    return SEQUENCE_INDICATORS.get(sequence_type, '[??]')

def is_mri_sequence(sequence_type: str) -> bool:
    return sequence_type in MRI_SEQUENCES

def is_ct_sequence(sequence_type: str) -> bool:
    return sequence_type in CT_SEQUENCES

def is_prostate_study_complete(available_sequences: Set[str]) -> bool:
    return PROSTATE_REQUIRED_SEQUENCES.issubset(available_sequences)

def get_missing_prostate_sequences(available_sequences: Set[str]) -> Set[str]:
    return PROSTATE_REQUIRED_SEQUENCES - available_sequences

