# GNNome 

A framework for training graph neural networks to untangle assembly graphs obtained from OLC-based de novo genome assemblers.



<p align="center">
  <img src="figures/GNNome.png" width="800" title="Framework">
</p>
Figure generated with DALL-E 3.

## Installation

### Requirements
- Linux (tested on Ubuntu 20.04)
- conda 4.6+
- CUDA 11.1+
- gcc 7.5+
- zlib 1.2.8+
- cmake 3.11+


### Setting up the environment

#### 1. Clone the repository
```bash
git clone https://github.com/lbcb-sci/GNNome.git
cd GNNome
```

#### 2. Create a conda virtual environment
```bash
conda create -n gnnome python=3.8 pip
conda activate gnnome
```

#### 2a. Install cmake and zlib
In case you don't already have them installed on your system you can install cmake and zlib inside your conda environment:
```bash
conda install cmake
conda install zlib
```

#### 2b. If you are using CUDA 12.0 or higher, install cudatoolkit v11.0
```bash
conda install cudatoolkit=11.0
```

#### 3. Install the requirements with pip (~3 min)

For GPU and CUDA 11.0+ run:
```bash
pip install -r requirements.txt
```
If you have no GPUs and no CUDA, we recommend running inference only. You can install the requirements with:
```bash
pip install -r requirements_cpu.txt
```

#### 4. Install tools used for constructing assembly graphs
```bash
python install_tools.py
```
This will install hifiasm and Raven which are used to generate HiFi and ONT assembly graphs, respectively. It also installs PBSIM which is used for simulating raw reads. All the tools are installed inside the `GNNome/vendor` directory.


<!-- #### 4a. Install hifiasm for constructing HiFi assembly graphs
This tool was tested with the hifiasm version 0.18.8:
```bash
git clone https://github.com/chhylp123/hifiasm.git --branch 0.18.8 --single-branch hifiasm-0.18.8
cd hifiasm-0.18.8
make
```

#### 4b. Install Raven for constructing ONT assembly graphs
This tool was tested with the Raven version 1.8.1:
```bash
git clone https://github.com/lbcb-sci/raven.git --branch print_graphs --single-branch raven-1.8.1
cd raven-1.8.1
cmake -S ./ -B./build -DRAVEN_BUILD_EXE=1 -DCMAKE_BUILD_TYPE=Release
cmake --build build
``` -->


## Example

The data needed to run the example consists of simulated E. coli reads (FASTA format) and an assembly graph of those reads generated with hifiasm (GFA format). Both can be found in the `example` directory. To run the example, there are three steps:

#### 1. Construct the assembly graph with hifiasm (<1 min)
```bash
mkdir -p example/hifiasm/output
./vendor/hifiasm-0.18.8/hifiasm --prt-raw -o example/hifiasm/output/ecoli_asm -t32 -l0 example/ecoli.fasta.gz
```

#### 2. Construct the neccesary data structures (DGL graphs and auxiliary dictionaries). (<1 min)
```bash
python create_inference_graphs.py --reads example/ecoli.fasta.gz --gfa example/hifiasm/output/ecoli_asm.bp.raw.r_utg.gfa --asm hifiasm --out example
```
The last command will create the following data inside the `example/hifiasm` directory.
- a DGL graph inside `example/hifiasm/processed` directory
- auxiliary data inside `example/hifiasm/info` directory

#### 3. Run the inference module. (<1 min)
```bash
python inference.py --data example --asm hifiasm --out example/hifiasm
```
The edge-probabilities will be computed with the deafult model (reported in the paper).
The directories `assembly`, `decode`, and `checkpoint` will be created inside `example/hifiasm`. You can find the assembly sequence in
`example/hifiasm/assembly/0_assembly.fasta`.




## Usage


### Easy way
If you want to just provide the reads and assemble the genome following the recommended pipeline (using hifiasm to build the assembly graph and the default model to untangle it), you can use the following command:
```bash
python run.py -r <reads> -o <out>
  -r <reads>
    input file in FASTA/FASTQ format (can be compressed with gzip)
  -o <out>
    path to where the processed data will be saved

  Optional:
  -t <threads>
    number of threads used for running the assembler (default: 1)
  -m <model>
    path to the model used for decoding (deafult: weights/weights.pt)
```

This will save the assembly to the path `<out>/hifiasm/assembly/0_assembly.fasta`. If you want more flexibility, e.g., where the data will be saved or which assembler you want to use, see the step-by-step instructions below.

### Step-by-step inference

To run the model on a new genome, first you need to run another assembler which can output a non-reduced graph in a GFA format.
Note: this tool has been optimized for haploid assembly, and this tutorial mainly focuses on this.


#### Construct the assembly graphs from HiFi sequences
For HiFi data, we recommend using [hifiasm](https://github.com/chhylp123/hifiasm).

Run hifiasm with the following command:
```bash
./vendor/hifiasm-0.18.8/hifiasm --prt-raw -o <out> -t <threads> -l0 <reads>
```
where `<reads>` is the path to the sequences in FASTA/FASTQ format, and `<out>` is the prefix for the output files. The GFA graph can then be found in the current directory under the name `<out>.bp.raw.r_utg.gfa`.

#### Construct the assembly graphs from ONT sequences
For ONT data, we recommend using [Raven](https://github.com/lbcb-sci/raven).

Run Raven with the following command:
```bash
./vendor/raven-1.8.1/builld/bin/raven -t <threads> -p0 <reads> > assembly.fasta
```
where `<reads>` is the path to the sequences in FASTA/FASTQ format.
The graph can then be found in the current directory under the name `graph_1.gfa`.


#### Process the assembly graphs

From the reads in FASTA/Q format and the graph in the GFA format, we can produce the graph in the DGL format and auxiliary data:
```bash
python create_inference_graphs.py --reads <reads> --gfa <gfa> --asm <asm> --out <out>

  <reads>
    input file in FASTA/FASTQ format (can be compressed with gzip)
  <gfa> 
    input file in GFA format
  <asm>
    assembler used for the assembly graph construction [hifiasm|raven]
  <out>
    path to where the processed data will be saved
```
The resulting data can be found in the `<out>/<asm>/processed/` and `<out>/<asm>/info/` directories.


#### Generating the assembly
```bash
python inference.py --data <data> --asm <asm> --out <out>

  <data>
    path to where the processed data is saved (same as <out> in the previous command)
  <asm>
    assembler used for the assembly graph construction [hifiasm|raven]
  <out>
    path to where the assembly will be saved
  
  optional:
    --model <model>
      path to the model used for decoding (deafult: weights/weights.pt)
```


### Training the network

#### Generate the training/validation data
You can generate synthetic training data by first simulating reads with PBSIM and then constructing assembly graphs with hifiasm or Raven. This consists of several steps.

Step 1. Specify which chromosomes you want to have in training and validation set, by editing values in the dictionaries in `train_valid_chrs.py`.

Step 2. Since the training is performed on individual chromosomes, you also need to have the sequences (references) of these chromosomes saved in a format `chr1.fasta`, `chr2.fasta`, etc. Full path to the directory where these chromosome references are stored is provided as an argument to the `generate_data.py` script (see below).

Step 3. PBSIM requires a sample profile files (e.g. `sample_pofile_ID.fastq` and `sample_pofile_ID.stats`) stored inside the `vendor/pbsim3` directory. You can download these files by running
```bash
bash download_profile.sh
```
The downloaded files correspond to the `sample_profile_ID` stated in the `config.py` dictionary. Alternatively, if you already have these files, copy them into `vendor/pbsim3` and edit the value of the dictionary in `config.py` under the key `sample_pofile_ID`. You can also create a new profile by editting values in the dictionary in `config.py` under the keys `sample_pofile_ID` and `sample_file`. Make sure to provide a unique ID for `sample_profile_ID`, and a path to an existing FASTQ file for `sample_file`. For more information, check [PBSIM3](https://github.com/yukiteruono/pbsim3).

Step 4. Finally, run the `generate_data.py` script:
```bash
python generate_data.py --datadir <datadir> --chrdir <chrdir> --asm <asm> --threads <threads>

  <datadir>
    path to directory where the generated data will be saved
  <chrdir>
    path to directory where the chromosome references are stored
  <asm>
    assembler used for the assembly graph construction [hifiasm|raven]
  <threads>
    number of threads used for running the assembler
```


#### Split the generated data into training and validation datasets.
Once the data has been generated and stored in the main database (the `<datadir>` that you provided in the previous step), you have to split it into training and validation datasets. This will copy data from the main database `<datadir>` into `<savedir>` (see below). Run the following command:

```bash
python split_data.py --datadir <datadir> --savedir <savedir> --name <name> --asm <asm>
  <datadir>
    path to directory where the generated data is saved
  <savedir>
    path to directory where the trainig/validation datasets will be copied
  <name>
    name assigned to the training and validation datasets
  <asm>
    assembler used for the assembly graph construction [hifiasm|raven]
```
Once all the data is copied, the script will print out the full paths of the training and validation directories. You can provide those paths as arguments to `train.py` script (see the next step).


#### Train the model
```bash
python train.py --train <train> --valid <valid> --asm <asm>
  
  <train>
    path to directory where the training data (provided by split_data.py)
  <valid>
    path to directory where the validation data (provided by split_data.py)
  <asm>
    assembler used to generate the training data [hifiasm|raven]

  optional:
    --name <name>
      name of the model that will be trained (default: date/time of execution)
    --overfit
      overfit on the training data
    --resume
      resume from a checkpoint, the <name> option has to be specified
    --dropout <dropout>
      dropout for training the model (default: 0)
    --seed <seed>
      seed for training the model (default: 1)
```

By default, the trained models and checkpointbs will be saved in the `models` and `checkpoints` directories, respectively. This can be changed in `config.py`. The name under which the model and checkpoint are saved is, by default, the timestamp of the run, if argument `--name` is not specified.


## Reproducibility
All the results in the paper can be reproduced by downloading the relevant data (link in the manuscript) and following the steps in the Usage section. Use the default weights for the model, available under `weights/weights.pt`.
