################################################################################

# Edit these three dictionaries to specify graphs to train/validation/test
# Assemblies will be constructed only for the graphs in the test_dict

# To train/validate/test on multiple chromosomes, put the as separate
# entries in the dictionaries
# E.g., to train on 1 chr19 graph and 2 chr20 graphs: 
# _train_dict = {'chr19': 1, 'chr20': 2}

# To test on real chromosome put "_r" suffix. Don't put value higher than 1,
# since there is only 1 real HiFi dataset for each chromosomes
# E.g., to test on real chr21:
# _test_dict = {'chr21_r': 1}

_train_dict = {
    'chr1_hg002':  15,
    'chr2_hg002':  0,
    'chr3_hg002':  15,
    'chr4_hg002':  0,
    'chr5_hg002':  15,
    'chr6_hg002':  0,
    'chr7_hg002':  0,
    'chr8_hg002':  0,
    'chr9_hg002':  15,
    'chr10_hg002': 0,
    'chr11_hg002': 0,
    'chr12_hg002': 15,
    'chr13_hg002': 0,
    'chr14_hg002': 0,
    'chr15_hg002': 0,
    'chr16_hg002': 0,
    'chr17_hg002': 0,
    'chr18_hg002': 15,
    'chr19_hg002': 0,
    'chr20_hg002': 0,
    'chr21_hg002': 0,
    'chr22_hg002': 0,
    'chrX_hg002':  0,
}

_valid_dict = {
    'chr1_hg002':  0,
    'chr2_hg002':  5,
    'chr3_hg002':  0,
    'chr4_hg002':  0,
    'chr5_hg002':  0,
    'chr6_hg002':  5,
    'chr7_hg002':  0,
    'chr8_hg002':  0,
    'chr9_hg002':  0,
    'chr10_hg002': 0,
    'chr11_hg002': 5,
    'chr12_hg002': 0,
    'chr13_hg002': 0,
    'chr14_hg002': 0,
    'chr15_hg002': 0,
    'chr16_hg002': 0,
    'chr17_hg002': 5,
    'chr18_hg002': 0,
    'chr19_hg002': 5,
    'chr20_hg002': 5,
    'chr21_hg002': 0,
    'chr22_hg002': 0,
    'chrX_hg002':  0,
}


def get_train_valid_chrs():
    return _train_dict, _valid_dict
