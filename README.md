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

#### 2.a Install cmake and zlib
In case you don't already have them installed on your system you can install cmake and zlib inside your conda environment:
```bash
conda install cmake
conda install zlib
```

#### 2.b If you are using CUDA 12.0 or higher, install cudatoolkit v11.0
```bash
conda install cudatoolkit=11.0
```

#### 3. Install the requirements with pip (~3 min)
```bash
pip install -r requirements.txt
```

#### 4a. Install hifiasm for constructing HiFi assembly graphs
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
```


## Example

The data needed to run the experiments consists of simulated E. coli reads (FASTA format) and an assembly graph of those reads generated with hifiasm (GFA format). Both can be found in the `example` directory. To pipeline consists of two steps

#### 1. Construct the assembly graph with hifiasm
```bash
mkdir -p example/hifiasm/output
./hifiasm-0.18.8/hifiasm --prt-raw -o example/hifiasm/output/ecoli_asm -t32 -l0 example/ecoli.fasta.gz
```

#### 2. Construct the neccesary data structures (DGL graphs and auxiliary dictionaries). (<1 min)
```bash
python create_inference_graphs.py --reads example/ecoli.fasta.gz --gfa example/hifiasm/output/ecoli_asm.bp.raw.r_utg.gfa --asm hifiasm --out example
```
This command will create the following data inside the `example/hifiasm` directory.
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

To run the model on a new genome, first you need to run another assembler which can output a non-reduced graph in a GFA format.
Note: this tool has been optimized for haploid assembly, and this tutorial mainly focuses on this.


### Construct the assembly graphs from HiFi sequences
For HiFi data, we recommend using [hifiasm](https://github.com/chhylp123/hifiasm).

Run hifiasm with the following command:
```bash
./hifiasm-0.18.8/hifiasm --prt-raw -o <out> -t <threads> -l0 <reads>
```
where `<reads>` is the path to the sequences in FASTA/FASTQ format, and `<out>` is the prefix for the output files. The GFA graph can then be found in the current directory under the name `<out>.bp.raw.r_utg.gfa`.

### Construct the assembly graphs from ONT sequences
For ONT data, we recommend using [Raven]().

Run Raven with the following command:
```bash
./raven-1.8.1/builld/bin/raven -t <threads> -p0 <reads> > assembly.fasta
```
where `<reads>` is the path to the sequences in FASTA/FASTQ format.
The graph can then be found in the current directory under the name `graph_1.gfa`.


### Process the assembly graphs

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


### Generating the assembly
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

#### Download the train/valid data
Link available in the manuscript, will be publicly available before acceptance

#### Train the model
```bash
python train.py --train <train> --valid <valid> --asm <asm>
  
  <train>
    Path to the training data
  <valid>
    Path to the validation data
  <asm>
    Assembler used to generate the training data [hifiasm|raven]

  optional:
    --name <name>
      Name of the model that will be trained (default: date/time of execution)
    --savedir <savedir>
      Name of the directory where the model and the checkpoints wiil be saved (default: checkpoints/)
    --overfit
      Overfit on the training data
    --resume
      Resume from a checkpoint, the <name> option has to be specified
    --dropout <dropout>
      Dropout for training the model (default: 0)
    --seed <seed>
      Seed for training the model (default: 1)
    
```

By default, the trained models will be saved in the `checkpoints` directory under name of today's date, if options `--name` and `--savedir` are not specified.

## Reproducibility
All the results in the paper can be reproduced by downloading the relevant data (link in the manuscript) and following the steps in the Usage section. Use the default weights for the model, available under `weights/weights.pt`.
