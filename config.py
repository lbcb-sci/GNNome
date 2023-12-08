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

_train_dict = {}
_valid_dict = {}

_train_dict_ont = {}
_valid_dict_ont = {}
_test_dict_ont  = {}


# Overfitting
# _train_dict = {'chr19_hg002': 1}
# _valid_dict = {'chr19_hg002': 0}
# _test_dict = {}

# _train_dict = {'chr1_ncbr': 1}
# _valid_dict = {}


    
# _train_dict = {'chr1_hg002': 30, 'chr9_hg002': 30, 'chr21_hg002': 0, 'chr19_hg002': 0, 'chr20_hg002': 0, 'chr22_hg002': 0, 'chr18_hg002': 30}
# _valid_dict = {'chr1_hg002': 0, 'chr9_hg002': 0, 'chr21_hg002': 0, 'chr19_hg002': 5, 'chr20_hg002': 5, 'chr22_hg002': 0, 'chr17_hg002': 5, 'chr11_hg002': 5}

# _train_dict['chr3_hg002'] = 30
# _train_dict['chr5_hg002'] = 30
# _train_dict['chr12_hg002'] = 30

# _valid_dict['chr6_hg002'] = 5


# Pretraining on synthetic HiFi data
# TRAIN
# _train_dict['chr1_hg002']  = 15
# _train_dict['chr3_hg002']  = 15
# _train_dict['chr5_hg002']  = 15
# _train_dict['chr9_hg002']  = 15
# _train_dict['chr12_hg002'] = 15
# _train_dict['chr18_hg002'] = 15

# _train_dict['chr1_arab'] = 10
# _train_dict['chr2_arab'] = 10
# _train_dict['chr3_arab'] = 10

_train_dict['chr1_zmays'] = 10
_train_dict['chr2_zmays']  = 10
_train_dict['chr3_zmays'] = 10
_train_dict['chr4_zmays']  = 10
_train_dict['chr5_zmays'] = 10
_train_dict['chr6_zmays']  = 10


# VALID
# _valid_dict['chr2_hg002']  = 10
# _valid_dict['chr6_hg002']  = 10
# _valid_dict['chr11_hg002'] = 10
# _valid_dict['chr17_hg002'] = 10
# _valid_dict['chr19_hg002'] = 10
# _valid_dict['chr20_hg002'] = 10

_valid_dict['chr7_zmays'] = 10
_valid_dict['chr8_zmays']  = 10
_valid_dict['chr9_zmays'] = 10
_valid_dict['chr10_zmays'] = 10

# _valid_dict['chr4_arab'] = 5
# _valid_dict['chr5_arab'] = 5


# Fine-tuning on real HiFi data
# TRAIN
# _train_dict['chr1_r']  = 1
# _train_dict['chr3_r']  = 1
# _train_dict['chr5_r']  = 1
# _train_dict['chr9_r']  = 1
# _train_dict['chr12_r'] = 1
# _train_dict['chr18_r'] = 1

# VALID
# _valid_dict['chr6_r']  = 1
# _valid_dict['chr11_r'] = 1
# _valid_dict['chr17_r'] = 1
# _valid_dict['chr19_r'] = 1
# _valid_dict['chr20_r'] = 1


# Training on ONT data
# TRAIN
_train_dict_ont['chr1_hg002']  = 15
_train_dict_ont['chr3_hg002']  = 15
_train_dict_ont['chr5_hg002']  = 15
_train_dict_ont['chr9_hg002']  = 15
_train_dict_ont['chr12_hg002'] = 15
_train_dict_ont['chr18_hg002'] = 15

# VALID
_valid_dict_ont['chr6_hg002']  = 5
_valid_dict_ont['chr11_hg002'] = 5
_valid_dict_ont['chr17_hg002'] = 5
_valid_dict_ont['chr19_hg002'] = 5
_valid_dict_ont['chr20_hg002'] = 5


# _train_dict['chr1_hg002+chr3_hg002+chr5_hg002+chr9_hg002+chr12_hg002+chr18_hg002'] = 15
# _train_dict['chr1_chm13+chr9_chm13+chr21_chm13'] = 5
# _valid_dict['chr6_hg002+chr11_hg002+chr17_hg002+chr19_hg002+chr20_hg002'] = 5


# _valid_dict['chrX_hg002'] = 5

# _train_dict['chr1_ncbr'] = 3
# _train_dict['chr2_ncbr'] = 3
# _train_dict['chr3_ncbr'] = 3
# _valid_dict['chr4_ncbr'] = 1
# _valid_dict['chr5_ncbr'] = 1
# _valid_dict['chr6_ncbr'] = 1
# _valid_dict['chr7_ncbr'] = 1

# _train_dict['chr1_hg002+chr9_hg002+chr18_hg002'] = 10
# _train_dict['chr1_chm13+chr9_chm13+chr21_chm13'] = 5
# _valid_dict['chr19_hg002+chr20_hg002+chr17_hg002'] = 3

# _train_dict = {'chr1_r': 1, 'chr9_r': 1, 'chr20_r': 1, 'chr1': 4, 'chr9': 4, 'chr20': 4}
# _valid_dict = {'chr19_r': 1, 'chr21_r': 1, 'chr22_r': 1,}

# Experiments
# _train_dict = {'chr1': 1, 'chr7': 1, 'chr9': 1, 'chr11': 1, 'chr17': 1, 'chr18': 1} 
# _valid_dict = {'chr10': 1, 'chr20': 1, 'chr22': 1}
_test_dict = {}


# _train_dict_ont = {'chr1_hg002': 5, 'chr9_hg002': 5, 'chr21_hg002': 0, 'chr19_hg002': 0, 'chr20_hg002': 0, 'chr22_hg002': 0, 'chr18_hg002': 5}
# _valid_dict_ont = {'chr1_hg002': 0, 'chr9_hg002': 0, 'chr21_hg002': 0, 'chr19_hg002': 2, 'chr20_hg002': 2, 'chr22_hg002': 0, 'chr17_hg002': 2}


# _train_dict = {}
# _valid_dict = {f'chr{i}': 1 for i in range(1, 23)} ; _valid_dict['chrX'] = 1
# _test_dict = {}
# _test_dict = {f'chr{i}': 1 for i in range(1, 23)} ; _test_dict['chrX'] = 1
################################################################################

def get_config():
    return {
        'train_dict': _train_dict,
        'valid_dict': _valid_dict,
        'test_dict' : _test_dict,
        'train_dict_ont': _train_dict_ont,
        'valid_dict_ont': _valid_dict_ont,
    }

