
# Edit these three dictionaries to specify graphs to train/validation/test

_train_dict = {
    'chr1_hg002':  0,
    'chr2_hg002':  0,
    'chr3_hg002':  0,
    'chr4_hg002':  0,
    'chr5_hg002':  0,
    'chr6_hg002':  0,
    'chr7_hg002':  0,
    'chr8_hg002':  0,
    'chr9_hg002':  0,
    'chr10_hg002': 0,
    'chr11_hg002': 0,
    'chr12_hg002': 0,
    'chr13_hg002': 0,
    'chr14_hg002': 0,
    'chr15_hg002': 0,
    'chr16_hg002': 0,
    'chr17_hg002': 0,
    'chr18_hg002': 1,
    'chr19_hg002': 0,
    'chr20_hg002': 1,
    'chr21_hg002': 0,
    'chr22_hg002': 0,
    'chrX_hg002':  0,
    # 'chr1_other': 0,
}

_valid_dict = {
    'chr1_hg002':  0,
    'chr2_hg002':  0,
    'chr3_hg002':  0,
    'chr4_hg002':  0,
    'chr5_hg002':  0,
    'chr6_hg002':  0,
    'chr7_hg002':  0,
    'chr8_hg002':  0,
    'chr9_hg002':  0,
    'chr10_hg002': 0,
    'chr11_hg002': 0,
    'chr12_hg002': 0,
    'chr13_hg002': 0,
    'chr14_hg002': 0,
    'chr15_hg002': 0,
    'chr16_hg002': 0,
    'chr17_hg002': 0,
    'chr18_hg002': 0,
    'chr19_hg002': 1,
    'chr20_hg002': 1,
    'chr21_hg002': 0,
    'chr22_hg002': 0,
    'chrX_hg002':  0,
    # 'chr1_other': 0,
}

def get_train_valid_chrs():
    return _train_dict, _valid_dict
