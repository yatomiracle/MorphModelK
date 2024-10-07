#!/bin/bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
if [ ! -d "NeuralMorphemeSegmentation" ] ; then
    git clone https://github.com/morozowdmitry/NeuralMorphemeSegmentation.git
fi
python mpe_morphemes.py