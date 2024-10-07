from NeuralMorphemeSegmentation.neural_morph_segm import load_cls
from pathlib import Path
import sys
import time

BASE_DIR = Path('__this__').parent.resolve()

PATHS = {
    'morphodict': BASE_DIR / 'models/morphodict_10_07_2023.json',
    'tikhonov': BASE_DIR / 'models/tikhonov_13_10_2023.json'
}

def predict(lemma, dataset_name):
    model = load_cls(PATHS[dataset_name])

    labels, _ = model._predict_probs([lemma])[0]
    morphemes, morpheme_types = model.labels_to_morphemes(
        lemma, labels, return_probs=False, return_types=True
    )

    parsing = [
        {"morpheme": morpheme, "type": morpheme_type}
        for morpheme, morpheme_type in zip(morphemes, morpheme_types)
    ]

    return parsing

def process_file(input_file, output_file, dataset_name):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            lemma = line.strip()
            if lemma:  # Проверяем, что строка не пустая
                result = predict(lemma, dataset_name)
                formatted_result = "\t".join([f"{m['morpheme']}:{m['type']}" for m in result])
                outfile.write(f"{lemma}\t{formatted_result}\n")

if __name__ == "__main__":
    start_time = time.time()

    if len(sys.argv) != 4:
        print("Usage: python mpe_morphemes.py <input_file> <output_file> <dataset_name>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    dataset_name = sys.argv[3]

    process_file(input_file, output_file, dataset_name)

    end_time = time.time()
    print(f"Processing time: {end_time - start_time} seconds")
