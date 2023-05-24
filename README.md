# Kurdish Llama
This is an attempt to fine-tune the Llama model released by Meta for Central Kurdish. The initial model was then fine-tuned on a set of instructions provided by Stanford's [Alpaca project](https://crfm.stanford.edu/2023/03/13/alpaca.html).

Another project, [GPT-4-LLM](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM), used the same set of instructions provided by the Alpaca project and generated output using `GPT-4` rather than the original `text-davinci-003`.

Most of the hard work has already been done. The goal of this project is to translate the dataset to Central Kurdish using an NLLB model. The resulting fine-tuned model, KurdishLlama, can be used for various natural language processing tasks in Central Kurdish.

Stay tuned for updates on the progress of this project!


### Translating the Dataset

To translate the dataset, run the following command:
```bash
python translate_data.py ./data/alpaca_gpt4_data.json ./data/alpaca_gpt4_ckb.json
```

This command will use an NLLB model to translate the Alpaca project's GPT-4 data to Central Kurdish, and save the translated data to a new file called `alpaca_gpt4_ckb.json`.


The NLLB model has a tendency to produce erroneous translations where it repeats a single word throughout. To address this issue, the `data_cleaning.py` script will remove any instances where a single word is repeated consecutively at least three times.



### Generation
To test the model you can either run the `generate.py` script or use the `Inference` notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hrazhan/kurdish-llama/blob/main/Inference.ipynb)
