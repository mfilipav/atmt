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

`(atmt) bs-mbpr113:preprocessed_data_bpe mfilipav$ subword-nmt learn-joint-bpe-and-vocab --input train.de train.en -s 16000 -o codes_file --write-vocabulary vocab_file.de vocab_file.en`
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

(atmt) bs-mbpr113:preprocessed_data_bpe mfilipav$ wc -l vocab_file.de
    7761 vocab_file.de
(atmt) bs-mbpr113:preprocessed_data_bpe mfilipav$ wc -l vocab_file.en
    5611 vocab_file.en
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
subword-nmt apply-bpe -c codes_file --vocabulary vocab_file.de --vocabulary-threshold 1 < train.de > train_bpe.de

subword-nmt apply-bpe -c codes_file --vocabulary vocab_file.en --vocabulary-threshold 1 < train.en > train_bpe.en
```

## contents of train_bpe.de
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


## Step 3: BPE encode dev and test bpe data, and tiny train
```
subword-nmt apply-bpe -c codes_file --vocabulary vocab_file.de --vocabulary-threshold 1 < test.de > test_bpe.de

subword-nmt apply-bpe -c codes_file --vocabulary vocab_file.en --vocabulary-threshold 1 < test.en > test_bpe.en

subword-nmt apply-bpe -c codes_file --vocabulary vocab_file.de --vocabulary-threshold 1 < valid.de > valid_bpe.de

subword-nmt apply-bpe -c codes_file --vocabulary vocab_file.en --vocabulary-threshold 1 < valid.en > valid_bpe.en

subword-nmt apply-bpe -c codes_file --vocabulary vocab_file.de --vocabulary-threshold 1 < tiny_train.de > tiny_train_bpe.de

subword-nmt apply-bpe -c codes_file --vocabulary vocab_file.en --vocabulary-threshold 1 < tiny_train.en > tiny_train_bpe.en
```


## [optional] Step 4: extract bpe vocab
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
  "is": 11,

python3 build_dictionary.py train_bpe.de train_bpe.en
gives out `train_bpe.de.json` and `train_bpe.en.json`


# Step 5: Pickle BPE encoded language files
TODO


