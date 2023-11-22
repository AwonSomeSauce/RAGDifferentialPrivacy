#!/bin/bash

# This stuff is required how much time you want to use hours:minutes:seconds
# Should try to minimize as much as possible because otherwise our usage goes up even though we are not using it
#SBATCH --time=24:00:00

# This is required as well
#SBATCH --account=def-nernst
# If you want to receive email updates when the job begins, ends, or fails

#SBATCH --mail-user=its.ahmed.musa@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# How much ram do you want, this would be 32G
#SBATCH --mem=64000M

# How many cpus do you need
#SBATCH --nodes=2

# How many gpus you want... these are the large v100s with 32G vram
#SBATCH --gres=gpu:v100l:2


# what modules you want to load.... scipy-stack contains a lot of basic stuff like pandas, numpy etc
# docs: https://docs.alliancecan.ca/wiki/Available_software
# module load cuda gcc
# module load arrow/11
# module load scipy-stack
module load python/3.10
module load gcc/9.3.0 arrow/11 cuda/11.4

# create virtual environment and install pip
virtualenv --no-download env
source env/bin/activate
pip install --no-index --upgrade pip

# pip install ./contractions-0.0.58-py2.py3-none-any.whl
pip install datasets transformers torch pandas numpy tqdm spacy sklearn scipy nltk presidio_analyzer presidio_anonymizer
# python -m spacy download en_core_web_lg

# Run your python file
#pip install --no-index -r requirements.txt
python main.py