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

```
INFO: Epoch 061: loss 2.212 | lr 0.0003 | num_tokens 9.794 | batch_size 1 | grad_norm 53.28 | clip 0.9988
INFO: Epoch 061: valid_loss 3.4 | num_tokens 10.2 | batch_size 500 | valid_perplexity 30
INFO: No validation set improvements observed for 3 epochs. Early stop!
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

## EXP1 Results
```
BLEU+case.mixed+numrefs.1+smooth.exp+tok.13a+version.1.4.14 = 17.2 49.8/21.9/12.1/6.9 (BP = 0.988 ratio = 0.988 hyp_len = 3968 ref_len = 4017)
```


# EXP2 BPE+dropout
## Training:
Here we show the last two epochs, 23 and 24:

```
*** Dropout applied to:  <_io.TextIOWrapper name='baseline/preprocessed_data_bpe_drop/train_bpe.de' mode='w' encoding='UTF-8'>
*** Dropout applied to:  <_io.TextIOWrapper name='baseline/preprocessed_data_bpe_drop/train_bpe.en' mode='w' encoding='UTF-8'>
Done BPE encoding with Dropout
[2020-11-08 11:39:30] COMMAND: preprocess.py --target-lang en --source-lang de --dest-dir baseline/prepared_data_bpe_drop/ --train-prefix baseline/preprocessed_data_bpe_drop/train_bpe --tiny-train-prefix baseline/preprocessed_data_bpe_drop/tiny_train_bpe --vocab-src baseline/preprocessed_data_bpe_drop/dict.de --vocab-trg baseline/preprocessed_data_bpe_drop/dict.en
[2020-11-08 11:39:30] Arguments: {'source_lang': 'de', 'target_lang': 'en', 'train_prefix': 'baseline/preprocessed_data_bpe_drop/train_bpe', 'tiny_train_prefix': 'baseline/preprocessed_data_bpe_drop/tiny_train_bpe', 'valid_prefix': None, 'test_prefix': None, 'dest_dir': 'baseline/prepared_data_bpe_drop/', 'threshold_src': 2, 'num_words_src': -1, 'threshold_tgt': 2, 'num_words_tgt': -1, 'vocab_src': 'baseline/preprocessed_data_bpe_drop/dict.de', 'vocab_trg': 'baseline/preprocessed_data_bpe_drop/dict.en'}
[2020-11-08 11:39:30] COMMAND: preprocess.py --target-lang en --source-lang de --dest-dir baseline/prepared_data_bpe_drop/ --train-prefix baseline/preprocessed_data_bpe_drop/train_bpe --tiny-train-prefix baseline/preprocessed_data_bpe_drop/tiny_train_bpe --vocab-src baseline/preprocessed_data_bpe_drop/dict.de --vocab-trg baseline/preprocessed_data_bpe_drop/dict.en
[2020-11-08 11:39:30] Arguments: {'source_lang': 'de', 'target_lang': 'en', 'train_prefix': 'baseline/preprocessed_data_bpe_drop/train_bpe', 'tiny_train_prefix': 'baseline/preprocessed_data_bpe_drop/tiny_train_bpe', 'valid_prefix': None, 'test_prefix': None, 'dest_dir': 'baseline/prepared_data_bpe_drop/', 'threshold_src': 2, 'num_words_src': -1, 'threshold_tgt': 2, 'num_words_tgt': -1, 'vocab_src': 'baseline/preprocessed_data_bpe_drop/dict.de', 'vocab_trg': 'baseline/preprocessed_data_bpe_drop/dict.en'}
[2020-11-08 11:39:30] Loaded a source dictionary (en) with 7764 words
[2020-11-08 11:39:30] Loaded a target dictionary (en) with 5614 words
[2020-11-08 11:39:32] Built a binary dataset for baseline/preprocessed_data_bpe_drop/train_bpe.de: 10000 sentences, 127780 tokens, 0.462% replaced by unknown token
[2020-11-08 11:39:33] Built a binary dataset for baseline/preprocessed_data_bpe_drop/tiny_train_bpe.de: 1000 sentences, 13100 tokens, 0.550% replaced by unknown token
[2020-11-08 11:39:35] Built a binary dataset for baseline/preprocessed_data_bpe_drop/train_bpe.en: 10000 sentences, 124516 tokens, 1.688% replaced by unknown token
[2020-11-08 11:39:35] Built a binary dataset for baseline/preprocessed_data_bpe_drop/tiny_train_bpe.en: 1000 sentences, 12662 tokens, 1.548% replaced by unknown token
Done Processing BPE files with dropout
INFO: Epoch 023: loss 3.479 | lr 0.0003 | num_tokens 12.45 | batch_size 1 | grad_norm 45.04 | clip 1
INFO: Epoch 023: valid_loss 3.78 | num_tokens 10.2 | batch_size 500 | valid_perplexity 43.7
*** Dropout applied to:  <_io.TextIOWrapper name='baseline/preprocessed_data_bpe_drop/train_bpe.de' mode='w' encoding='UTF-8'>
*** Dropout applied to:  <_io.TextIOWrapper name='baseline/preprocessed_data_bpe_drop/train_bpe.en' mode='w' encoding='UTF-8'>
Done BPE encoding with Dropout
[2020-11-08 11:47:03] COMMAND: preprocess.py --target-lang en --source-lang de --dest-dir baseline/prepared_data_bpe_drop/ --train-prefix baseline/preprocessed_data_bpe_drop/train_bpe --tiny-train-prefix baseline/preprocessed_data_bpe_drop/tiny_train_bpe --vocab-src baseline/preprocessed_data_bpe_drop/dict.de --vocab-trg baseline/preprocessed_data_bpe_drop/dict.en
[2020-11-08 11:47:03] Arguments: {'source_lang': 'de', 'target_lang': 'en', 'train_prefix': 'baseline/preprocessed_data_bpe_drop/train_bpe', 'tiny_train_prefix': 'baseline/preprocessed_data_bpe_drop/tiny_train_bpe', 'valid_prefix': None, 'test_prefix': None, 'dest_dir': 'baseline/prepared_data_bpe_drop/', 'threshold_src': 2, 'num_words_src': -1, 'threshold_tgt': 2, 'num_words_tgt': -1, 'vocab_src': 'baseline/preprocessed_data_bpe_drop/dict.de', 'vocab_trg': 'baseline/preprocessed_data_bpe_drop/dict.en'}
[2020-11-08 11:47:03] COMMAND: preprocess.py --target-lang en --source-lang de --dest-dir baseline/prepared_data_bpe_drop/ --train-prefix baseline/preprocessed_data_bpe_drop/train_bpe --tiny-train-prefix baseline/preprocessed_data_bpe_drop/tiny_train_bpe --vocab-src baseline/preprocessed_data_bpe_drop/dict.de --vocab-trg baseline/preprocessed_data_bpe_drop/dict.en
[2020-11-08 11:47:03] Arguments: {'source_lang': 'de', 'target_lang': 'en', 'train_prefix': 'baseline/preprocessed_data_bpe_drop/train_bpe', 'tiny_train_prefix': 'baseline/preprocessed_data_bpe_drop/tiny_train_bpe', 'valid_prefix': None, 'test_prefix': None, 'dest_dir': 'baseline/prepared_data_bpe_drop/', 'threshold_src': 2, 'num_words_src': -1, 'threshold_tgt': 2, 'num_words_tgt': -1, 'vocab_src': 'baseline/preprocessed_data_bpe_drop/dict.de', 'vocab_trg': 'baseline/preprocessed_data_bpe_drop/dict.en'}
[2020-11-08 11:47:03] Loaded a source dictionary (en) with 7764 words
[2020-11-08 11:47:04] Loaded a target dictionary (en) with 5614 words
[2020-11-08 11:47:06] Built a binary dataset for baseline/preprocessed_data_bpe_drop/train_bpe.de: 10000 sentences, 127521 tokens, 0.460% replaced by unknown token
[2020-11-08 11:47:06] Built a binary dataset for baseline/preprocessed_data_bpe_drop/tiny_train_bpe.de: 1000 sentences, 13100 tokens, 0.550% replaced by unknown token
[2020-11-08 11:47:08] Built a binary dataset for baseline/preprocessed_data_bpe_drop/train_bpe.en: 10000 sentences, 124570 tokens, 1.612% replaced by unknown token
[2020-11-08 11:47:08] Built a binary dataset for baseline/preprocessed_data_bpe_drop/tiny_train_bpe.en: 1000 sentences, 12662 tokens, 1.548% replaced by unknown token
Done Processing BPE files with dropout
INFO: Epoch 024: loss 3.463 | lr 0.0003 | num_tokens 12.46 | batch_size 1 | grad_norm 45.03 | clip 1
INFO: Epoch 024: valid_loss 3.77 | num_tokens 10.2 | batch_size 500 | valid_perplexity 43.2
INFO: No validation set improvements observed for 3 epochs. Early stop!

```

## Translate EXP2

```
(atmt) bs-mbpr113:atmt mfilipav$ python3 translate.py --data baseline/prepared_data_bpe_drop --checkpoint-path checkpoints/checkpoint_exp2_bpe_dropout/checkpoint_best.pt --output translations/translation_exp2_bpe_dropout/model_translations.txt
[2020-11-08 15:34:21] COMMAND: translate.py --data baseline/prepared_data_bpe_drop --checkpoint-path checkpoints/checkpoint_exp2_bpe_dropout/checkpoint_best.pt --output translations/translation_exp2_bpe_dropout/model_translations.txt
[2020-11-08 15:34:21] Arguments: {'cuda': False, 'seed': 42, 'data': 'baseline/prepared_data_bpe_drop', 'checkpoint_path': 'checkpoints/checkpoint_exp2_bpe_dropout/checkpoint_best.pt', 'batch_size': 1, 'output': 'translations/translation_exp2_bpe_dropout/model_translations.txt', 'max_len': 25, 'source_lang': 'de', 'target_lang': 'en', 'max_tokens': None, 'train_on_tiny': False, 'arch': 'lstm', 'max_epoch': 100, 'clip_norm': 4.0, 'lr': 0.0003, 'patience': 3, 'log_file': None, 'save_dir': 'checkpoints/checkpoint_exp2_bpe_dropout', 'restore_file': 'checkpoint_last.pt', 'save_interval': 1, 'no_save': False, 'epoch_checkpoints': False, 'encoder_embed_dim': 64, 'encoder_embed_path': None, 'encoder_hidden_size': 64, 'encoder_num_layers': 1, 'encoder_bidirectional': 'True', 'encoder_dropout_in': 0.25, 'encoder_dropout_out': 0.25, 'decoder_embed_dim': 64, 'decoder_embed_path': None, 'decoder_hidden_size': 128, 'decoder_num_layers': 1, 'decoder_dropout_in': 0.25, 'decoder_dropout_out': 0.25, 'decoder_use_attention': 'True', 'decoder_use_lexical_model': 'False', 'device_id': 0}
[2020-11-08 15:34:21] Loaded a source dictionary (de) with 7764 words
[2020-11-08 15:34:21] Loaded a target dictionary (en) with 5614 words
[2020-11-08 15:34:21] Loaded a model from checkpoint checkpoints/checkpoint_exp2_bpe_dropout/checkpoint_best.pt
```



## Re-process and Evaluate EXP2 with sacrebleau
```
python3 translate.py --data baseline/prepared_data_bpe_drop --checkpoint-path checkpoints/checkpoint_exp2_bpe_dropout/checkpoint_best.pt --output translations/translation_exp2_bpe_dropout/model_translations.txt
[2020-11-08 15:34:21] COMMAND: translate.py --data baseline/prepared_data_bpe_drop --checkpoint-path checkpoints/checkpoint_exp2_bpe_dropout/checkpoint_best.pt --output translations/translation_exp2_bpe_dropout/model_translations.txt
[2020-11-08 15:34:21] Arguments: {'cuda': False, 'seed': 42, 'data': 'baseline/prepared_data_bpe_drop', 'checkpoint_path': 'checkpoints/checkpoint_exp2_bpe_dropout/checkpoint_best.pt', 'batch_size': 1, 'output': 'translations/translation_exp2_bpe_dropout/model_translations.txt', 'max_len': 25, 'source_lang': 'de', 'target_lang': 'en', 'max_tokens': None, 'train_on_tiny': False, 'arch': 'lstm', 'max_epoch': 100, 'clip_norm': 4.0, 'lr': 0.0003, 'patience': 3, 'log_file': None, 'save_dir': 'checkpoints/checkpoint_exp2_bpe_dropout', 'restore_file': 'checkpoint_last.pt', 'save_interval': 1, 'no_save': False, 'epoch_checkpoints': False, 'encoder_embed_dim': 64, 'encoder_embed_path': None, 'encoder_hidden_size': 64, 'encoder_num_layers': 1, 'encoder_bidirectional': 'True', 'encoder_dropout_in': 0.25, 'encoder_dropout_out': 0.25, 'decoder_embed_dim': 64, 'decoder_embed_path': None, 'decoder_hidden_size': 128, 'decoder_num_layers': 1, 'decoder_dropout_in': 0.25, 'decoder_dropout_out': 0.25, 'decoder_use_attention': 'True', 'decoder_use_lexical_model': 'False', 'device_id': 0}
[2020-11-08 15:34:21] Loaded a source dictionary (de) with 7764 words
[2020-11-08 15:34:21] Loaded a target dictionary (en) with 5614 words
[2020-11-08 15:34:21] Loaded a model from checkpoint checkpoints/checkpoint_exp2_bpe_dropout/checkpoint_best.pt
(atmt) bs-mbpr113:atmt mfilipav$ bash postprocess.sh translations/translation_exp2_bpe_dropout/model_translations.txt translations/translation_exp2_bpe_dropout/model_translations.out en
(atmt) bs-mbpr113:atmt mfilipav$ cat translations/translation_exp2_bpe_dropout/model_translations.out | sacrebleu baseline/raw_data/test.en

checkpoint_best:
BLEU+case.mixed+numrefs.1+smooth.exp+tok.13a+version.1.4.14 = 7.4 
36.4/10.7/4.1/1.9 (BP = 1.000 ratio = 1.135 hyp_len = 4558 ref_len = 4017)

checkpoint_last:
BLEU+case.mixed+numrefs.1+smooth.exp+tok.13a+version.1.4.14 = 8.0 36.3/11.4/4.6/2.1 (BP = 1.000 ratio = 1.154 hyp_len = 4637 ref_len = 4017)

```


# Compare the reference, and EXP1 and EXP2 translations

```
Reference:
    1. We've already met.
    2. Jumping rope is my daughter's favorite.
    3. If you go fishing tomorrow, I will, too.
    4. Tom has to look after Mary.
    5. I want you to go with Tom.
    6. Safety is the primary concern.

EXP1: BPE
    1. We have to meet us.
    2. The daughter is to be love with my daughter.
    3. If you've been going to go to the same, I'll get.
    4. Tom must be able to Mary.
    5. I want to tell you Tom.
    6. The work is the only one of the same one.

EXP2: BPE dropout
    1. We have to do us.
    2. The li@@ k@@ e is my fa@@ ther.
    3. If you should go to Boston, I am.
    4. Tom must be at Mary.
    5. I want you to do Tom.
    6. The g@@ oo@@ d is the best of the b@@ oo@@ k.
```

