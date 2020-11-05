# ATMT Assignment 3: improving a low-resource NMT system
My solution for this assignment.

Improvements could be:
* BPE algorithm implementation
* BPE Dropout
* Source Copying
* Hyperparam tuning

# atmt code base
Materials for the first assignment of "Advanced Techniques of Machine Translation".

Please refer to the assignment sheet for instructions on how to use the toolkit.

The toolkit is based on [this implementation](https://github.com/demelin/nmt_toolkit).




EXPERIMENT 1: BPE ENCODING
=============================

# Data preparation with subword-nmt
Follow steps from `https://github.com/rsennrich/subword-nmt#best-practice-advice-for-byte-pair-encoding-in-nmt` to prepare datasets.

Our starting point is files in `baseline/preprocessed_data`, while preprocessing output is in `baseline/preprocessed_data_bpe`

## Some explanations of flags:
```
    - codes: help="File with BPE codes (created by learn_bpe.py).")
    - merges: help="Use this many BPE operations (<= number of learned symbols)"+
                 "default: Apply all the learned merge operations")
    - s or separator: help="Separator between non-final subword units (default: '%(default)s'))")
    - vocabulary: help="Vocabulary file (built with get_vocab.py). If provided, this script reverts any merge operations that produce an OOV.")
    - vocabulary_thershold: help="Vocabulary threshold. If vocabulary is provided, any word with frequency < threshold will be treated as OOV")
```

## Step 1: learn a joint BPE vocab
Use byte pair encoding (BPE) to learn a variable-length encoding of the vocabulary in a text.
This script learns BPE jointly on a concatenation of a list of texts (typically the source and target side of a parallel corpus,
applies the learned operation to each and (optionally) returns the resulting vocabulary of each text.
The vocabulary can be used in apply_bpe.py to avoid producing symbols that are rare or OOV in a training text.

`(atmt) bs-mbpr113:preprocessed_data_bpe mfilipav$ subword-nmt learn-joint-bpe-and-vocab --input train.de train.en -s 16000 -o codes_file --write-vocabulary dict.de dict.en`
13451 codes_file
```#version: 0.2
e n</w>
c h
i n
e r</w>
e r
t h
c h</w>
...
...

14 92</w>
1 7
1 6</w>
1 30</w>
1 2</w>
1 23</w>
1 000</w>
1 00
100 3</w>
1 .
1. 3</w>
000 .
000 ,
. S
```

(atmt) bs-mbpr113:preprocessed_data_bpe mfilipav$ wc -l dict.de
    7761 dict.de
(atmt) bs-mbpr113:preprocessed_data_bpe mfilipav$ wc -l dict.en
    5611 dict.en
- vocab files have token frequencies
```
. 8230
, 3268
ich 2489
Tom 2044
ist 1562
? 1500
nicht 1366
das 1226
die 1110
du 1085
zu 981
...
...
Kam@@ 1
Aoi 1
wunden 1
41 1
kne@@ 1
schlossen 1
Mittel@@ 1
```

```
. 8559
the 2698
I 2572
you 2254
to 2193
Tom 2109
a 1648
? 1499
...
...
Hermann 1
Hesse 1
quot@@ 1
Aoi 1
ten@@ 1
41 1
stom@@ 1
che 1
```

## Step 2: BPE encode train data with 
```
subword-nmt apply-bpe -c codes_file --vocabulary dict.de --vocabulary-threshold 1 < train.de > train_bpe.de

subword-nmt apply-bpe -c codes_file --vocabulary dict.en --vocabulary-threshold 1 < train.en > train_bpe.en
```

Contents of train_bpe.de
```
    Tom wurde von der Polizei inhaftiert .
    dein Name kommt mir bekannt vor .
    es wurde auf 10@@ .@@ 000 Yen er@@ rechnet .
    heute habe ich viel zu tun .
    ich habe ein Jahr in Boston studiert .
    es sieht nicht gut aus für Tom .
    er geht nicht mehr zur Schule .
    ich trau@@ te meinen Ohren nicht !

    Tom wurde von der Polizei inhaftiert .
    dein Name kommt mir bekannt vor .
    es wurde auf 10.000 Yen errechnet .
    heute habe ich viel zu tun .
    ich habe ein Jahr in Boston studiert .
    es sieht nicht gut aus für Tom .
    er geht nicht mehr zur Schule .
    ich traute meinen Ohren nicht !
```

## Step 3: BPE encode dev and test bpe data, and tiny train
```
subword-nmt apply-bpe -c codes_file --vocabulary dict.de --vocabulary-threshold 1 < test.de > test_bpe.de

subword-nmt apply-bpe -c codes_file --vocabulary dict.en --vocabulary-threshold 1 < test.en > test_bpe.en

subword-nmt apply-bpe -c codes_file --vocabulary dict.de --vocabulary-threshold 1 < valid.de > valid_bpe.de

subword-nmt apply-bpe -c codes_file --vocabulary dict.en --vocabulary-threshold 1 < valid.en > valid_bpe.en

subword-nmt apply-bpe -c codes_file --vocabulary dict.de --vocabulary-threshold 1 < tiny_train.de > tiny_train_bpe.de

subword-nmt apply-bpe -c codes_file --vocabulary dict.en --vocabulary-threshold 1 < tiny_train.en > tiny_train_bpe.en
```


## [optional] Step 4: extract bpe vocab

`python3 build_dictionary.py train_bpe.de train_bpe.en`
gives out `train_bpe.de.json` and `train_bpe.en.json`

```
    {
      "<EOS>": 0,
      "<GO>": 1,
      "<UNK>": 2,
      ".": 3,
      "the": 4,
      "I": 5,
      "you": 6,
      "to": 7,
      "Tom": 8,
      "a": 9,
      "?": 10,
      "is": 11,...
```



## Step 5: Pickle BPE encoded language files
Use `preprocess_bpe_data.sh` script

```
With 'threshold_src': 1, 'num_words_src': 4000, 'threshold_tgt': 1, 'num_words_tgt': 4000:

(atmt) bs-mbpr113:atmt mfilipav$ bash preprocess_bpe_data.sh 
[2020-11-05 12:22:11] COMMAND: preprocess.py --target-lang en --source-lang de --dest-dir baseline/prepared_data_bpe/ --train-prefix baseline/preprocessed_data_bpe/train_bpe --valid-prefix baseline/preprocessed_data_bpe/valid_bpe --test-prefix baseline/preprocessed_data_bpe/test_bpe --tiny-train-prefix baseline/preprocessed_data_bpe/tiny_train_bpe --threshold-src 1 --threshold-tgt 1 --num-words-src 4000 --num-words-tgt 4000
[2020-11-05 12:22:11] Arguments: {'source_lang': 'de', 'target_lang': 'en', 'train_prefix': 'baseline/preprocessed_data_bpe/train_bpe', 'tiny_train_prefix': 'baseline/preprocessed_data_bpe/tiny_train_bpe', 'valid_prefix': 'baseline/preprocessed_data_bpe/valid_bpe', 'test_prefix': 'baseline/preprocessed_data_bpe/test_bpe', 'dest_dir': 'baseline/prepared_data_bpe/', 'threshold_src': 1, 'num_words_src': 4000, 'threshold_tgt': 1, 'num_words_tgt': 4000, 'vocab_src': None, 'vocab_trg': None}
[2020-11-05 12:22:11] COMMAND: preprocess.py --target-lang en --source-lang de --dest-dir baseline/prepared_data_bpe/ --train-prefix baseline/preprocessed_data_bpe/train_bpe --valid-prefix baseline/preprocessed_data_bpe/valid_bpe --test-prefix baseline/preprocessed_data_bpe/test_bpe --tiny-train-prefix baseline/preprocessed_data_bpe/tiny_train_bpe --threshold-src 1 --threshold-tgt 1 --num-words-src 4000 --num-words-tgt 4000
[2020-11-05 12:22:11] Arguments: {'source_lang': 'de', 'target_lang': 'en', 'train_prefix': 'baseline/preprocessed_data_bpe/train_bpe', 'tiny_train_prefix': 'baseline/preprocessed_data_bpe/tiny_train_bpe', 'valid_prefix': 'baseline/preprocessed_data_bpe/valid_bpe', 'test_prefix': 'baseline/preprocessed_data_bpe/test_bpe', 'dest_dir': 'baseline/prepared_data_bpe/', 'threshold_src': 1, 'num_words_src': 4000, 'threshold_tgt': 1, 'num_words_tgt': 4000, 'vocab_src': None, 'vocab_trg': None}
[2020-11-05 12:22:11] Built a source dictionary (de) with 4000 words
[2020-11-05 12:22:11] Built a target dictionary (en) with 4000 words
[2020-11-05 12:22:13] Built a binary dataset for baseline/preprocessed_data_bpe/train_bpe.de: 10000 sentences, 101424 tokens, 5.774% replaced by unknown token
[2020-11-05 12:22:13] Built a binary dataset for baseline/preprocessed_data_bpe/tiny_train_bpe.de: 1000 sentences, 10686 tokens, 5.690% replaced by unknown token
[2020-11-05 12:22:14] Built a binary dataset for baseline/preprocessed_data_bpe/valid_bpe.de: 500 sentences, 5339 tokens, 5.900% replaced by unknown token
[2020-11-05 12:22:14] Built a binary dataset for baseline/preprocessed_data_bpe/test_bpe.de: 500 sentences, 5337 tokens, 5.565% replaced by unknown token
[2020-11-05 12:22:15] Built a binary dataset for baseline/preprocessed_data_bpe/train_bpe.en: 10000 sentences, 97944 tokens, 1.929% replaced by unknown token
[2020-11-05 12:22:16] Built a binary dataset for baseline/preprocessed_data_bpe/tiny_train_bpe.en: 1000 sentences, 10159 tokens, 2.254% replaced by unknown token
[2020-11-05 12:22:16] Built a binary dataset for baseline/preprocessed_data_bpe/valid_bpe.en: 500 sentences, 5100 tokens, 2.216% replaced by unknown token
[2020-11-05 12:22:16] Built a binary dataset for baseline/preprocessed_data_bpe/test_bpe.en: 500 sentences, 5134 tokens, 2.552% replaced by unknown token



With --vocab-src baseline/preprocessed_data_bpe/dict.de --vocab-trg baseline/preprocessed_data_bpe/dict.en:

[2020-11-05 12:22:16] COMMAND: preprocess.py --target-lang en --source-lang de --dest-dir baseline/prepared_data_bpe/ --train-prefix baseline/preprocessed_data_bpe/train_bpe --valid-prefix baseline/preprocessed_data_bpe/valid_bpe --test-prefix baseline/preprocessed_data_bpe/test_bpe --tiny-train-prefix baseline/preprocessed_data_bpe/tiny_train_bpe --vocab-src baseline/preprocessed_data_bpe/dict.de --vocab-trg baseline/preprocessed_data_bpe/dict.en
[2020-11-05 12:22:16] Arguments: {'source_lang': 'de', 'target_lang': 'en', 'train_prefix': 'baseline/preprocessed_data_bpe/train_bpe', 'tiny_train_prefix': 'baseline/preprocessed_data_bpe/tiny_train_bpe', 'valid_prefix': 'baseline/preprocessed_data_bpe/valid_bpe', 'test_prefix': 'baseline/preprocessed_data_bpe/test_bpe', 'dest_dir': 'baseline/prepared_data_bpe/', 'threshold_src': 2, 'num_words_src': -1, 'threshold_tgt': 2, 'num_words_tgt': -1, 'vocab_src': 'baseline/preprocessed_data_bpe/dict.de', 'vocab_trg': 'baseline/preprocessed_data_bpe/dict.en'}
[2020-11-05 12:22:16] COMMAND: preprocess.py --target-lang en --source-lang de --dest-dir baseline/prepared_data_bpe/ --train-prefix baseline/preprocessed_data_bpe/train_bpe --valid-prefix baseline/preprocessed_data_bpe/valid_bpe --test-prefix baseline/preprocessed_data_bpe/test_bpe --tiny-train-prefix baseline/preprocessed_data_bpe/tiny_train_bpe --vocab-src baseline/preprocessed_data_bpe/dict.de --vocab-trg baseline/preprocessed_data_bpe/dict.en
[2020-11-05 12:22:16] Arguments: {'source_lang': 'de', 'target_lang': 'en', 'train_prefix': 'baseline/preprocessed_data_bpe/train_bpe', 'tiny_train_prefix': 'baseline/preprocessed_data_bpe/tiny_train_bpe', 'valid_prefix': 'baseline/preprocessed_data_bpe/valid_bpe', 'test_prefix': 'baseline/preprocessed_data_bpe/test_bpe', 'dest_dir': 'baseline/prepared_data_bpe/', 'threshold_src': 2, 'num_words_src': -1, 'threshold_tgt': 2, 'num_words_tgt': -1, 'vocab_src': 'baseline/preprocessed_data_bpe/dict.de', 'vocab_trg': 'baseline/preprocessed_data_bpe/dict.en'}
[2020-11-05 12:22:16] Loaded a source dictionary (en) with 7764 words
[2020-11-05 12:22:16] Loaded a target dictionary (en) with 5614 words
[2020-11-05 12:22:18] Built a binary dataset for baseline/preprocessed_data_bpe/train_bpe.de: 10000 sentences, 101424 tokens, 0.000% replaced by unknown token
[2020-11-05 12:22:18] Built a binary dataset for baseline/preprocessed_data_bpe/tiny_train_bpe.de: 1000 sentences, 10686 tokens, 0.131% replaced by unknown token
[2020-11-05 12:22:18] Built a binary dataset for baseline/preprocessed_data_bpe/valid_bpe.de: 500 sentences, 5339 tokens, 0.075% replaced by unknown token
[2020-11-05 12:22:18] Built a binary dataset for baseline/preprocessed_data_bpe/test_bpe.de: 500 sentences, 5337 tokens, 0.150% replaced by unknown token
[2020-11-05 12:22:20] Built a binary dataset for baseline/preprocessed_data_bpe/train_bpe.en: 10000 sentences, 97944 tokens, 0.000% replaced by unknown token
[2020-11-05 12:22:20] Built a binary dataset for baseline/preprocessed_data_bpe/tiny_train_bpe.en: 1000 sentences, 10159 tokens, 0.108% replaced by unknown token
[2020-11-05 12:22:20] Built a binary dataset for baseline/preprocessed_data_bpe/valid_bpe.en: 500 sentences, 5100 tokens, 0.118% replaced by unknown token
[2020-11-05 12:22:20] Built a binary dataset for baseline/preprocessed_data_bpe/test_bpe.en: 500 sentences, 5134 tokens, 0.019% replaced by unknown token

```

What we used:
```
[2020-11-05 12:25:17] Loaded a source dictionary (en) with 7764 words
[2020-11-05 12:25:17] Loaded a target dictionary (en) with 5614 words
[2020-11-05 12:25:19] Built a binary dataset for baseline/preprocessed_data_bpe/train_bpe.de: 10000 sentences, 101424 tokens, 0.000% replaced by unknown token
[2020-11-05 12:25:19] Built a binary dataset for baseline/preprocessed_data_bpe/tiny_train_bpe.de: 1000 sentences, 10686 tokens, 0.131% replaced by unknown token
[2020-11-05 12:25:19] Built a binary dataset for baseline/preprocessed_data_bpe/valid_bpe.de: 500 sentences, 5339 tokens, 0.075% replaced by unknown token
[2020-11-05 12:25:19] Built a binary dataset for baseline/preprocessed_data_bpe/test_bpe.de: 500 sentences, 5337 tokens, 0.150% replaced by unknown token
[2020-11-05 12:25:21] Built a binary dataset for baseline/preprocessed_data_bpe/train_bpe.en: 10000 sentences, 97944 tokens, 0.000% replaced by unknown token
[2020-11-05 12:25:22] Built a binary dataset for baseline/preprocessed_data_bpe/tiny_train_bpe.en: 1000 sentences, 10159 tokens, 0.108% replaced by unknown token
[2020-11-05 12:25:22] Built a binary dataset for baseline/preprocessed_data_bpe/valid_bpe.en: 500 sentences, 5100 tokens, 0.118% replaced by unknown token
[2020-11-05 12:25:22] Built a binary dataset for baseline/preprocessed_data_bpe/test_bpe.en: 500 sentences, 5134 tokens, 0.019% replaced by unknown token
```


## Step 6: add `dict_<lang>` to `baseline/prepared_data_bpe/dict.<lang>`





# Training, Translating, Raw texting, sacreBLEU
## Step 1:
```
python3 train.py --data baseline/prepared_data_bpe --source-lang de --target-lang en --batch-size 1 --max-epoch 100 --patience 3 --save-dir checkpoints/checkpoint_exp1_bpe
```

## Step 2 Translate from DE to EN
```
python3 translate.py --data baseline/prepared_data_bpe --checkpoint-path checkpoints/checkpoint_exp1_bpe/checkpoint_ --output translations/translation_exp1_bpe/model_translations.txt
```

[2020-11-05 17:24:44] Loaded a source dictionary (de) with 7764 words
[2020-11-05 17:24:44] Loaded a target dictionary (en) with 5614 words
[2020-11-05 17:24:44] Loaded a model from checkpoint checkpoints/checkpoint_exp1_bpe/checkpoint_best.pt


## Step 3 Reverse pre-processing and Eval with SacredBleu
```
bash postprocess.sh translations/translation_exp1_bpe/model_translations.txt translations/translation_exp1_bpe/model_translations.out en

cat translations/translation_exp1_bpe/model_translations.out | sacrebleu baseline/raw_data/test.en
```

Dummy results:
`BLEU+case.mixed+numrefs.1+smooth.exp+tok.13a+version.1.4.14 = 0.3 28.6/0.8/0.1/0.0 (BP = 0.686 ratio = 0.726 hyp_len = 2916 ref_len = 4017)`
