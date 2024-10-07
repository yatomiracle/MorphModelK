from NeuralMorphemeSegmentation.neural_morph_segm import load_cls

from pathlib import Path


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


if __name__ == "__main__":
    print(predict(lemma="слово", dataset_name='morphodict'))
