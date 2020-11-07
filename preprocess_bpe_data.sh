# cat baseline/raw_data/train.de | perl moses_scripts/normalize-punctuation.perl -l de | perl moses_scripts/tokenizer.perl -l de -a -q > baseline/preprocessed_data/train.de.p

# cat baseline/raw_data/train.en | perl moses_scripts/normalize-punctuation.perl -l en | perl moses_scripts/tokenizer.perl -l en -a -q > baseline/preprocessed_data/train.en.p

# perl moses_scripts/train-truecaser.perl --model baseline/preprocessed_data/tm.de --corpus baseline/preprocessed_data/train.de.p

# perl moses_scripts/train-truecaser.perl --model baseline/preprocessed_data/tm.en --corpus baseline/preprocessed_data/train.en.p

# cat baseline/preprocessed_data/train.de.p | perl moses_scripts/truecase.perl --model baseline/preprocessed_data/tm.de > baseline/preprocessed_data/train.de 

# cat baseline/preprocessed_data/train.en.p | perl moses_scripts/truecase.perl --model baseline/preprocessed_data/tm.en > baseline/preprocessed_data/train.en

# cat baseline/raw_data/valid.de | perl moses_scripts/normalize-punctuation.perl -l de | perl moses_scripts/tokenizer.perl -l de -a -q | perl moses_scripts/truecase.perl --model baseline/preprocessed_data/tm.de > baseline/preprocessed_data/valid.de

# cat baseline/raw_data/valid.en | perl moses_scripts/normalize-punctuation.perl -l en | perl moses_scripts/tokenizer.perl -l en -a -q | perl moses_scripts/truecase.perl --model baseline/preprocessed_data/tm.en > baseline/preprocessed_data/valid.en

# cat baseline/raw_data/test.de | perl moses_scripts/normalize-punctuation.perl -l de | perl moses_scripts/tokenizer.perl -l de -a -q | perl moses_scripts/truecase.perl --model baseline/preprocessed_data/tm.de > baseline/preprocessed_data/test.de

# cat baseline/raw_data/test.en | perl moses_scripts/normalize-punctuation.perl -l en | perl moses_scripts/tokenizer.perl -l en -a -q | perl moses_scripts/truecase.perl --model baseline/preprocessed_data/tm.en > baseline/preprocessed_data/test.en

# cat baseline/raw_data/tiny_train.de | perl moses_scripts/normalize-punctuation.perl -l de | perl moses_scripts/tokenizer.perl -l de -a -q | perl moses_scripts/truecase.perl --model baseline/preprocessed_data/tm.de > baseline/preprocessed_data/tiny_train.de

# cat baseline/raw_data/tiny_train.en | perl moses_scripts/normalize-punctuation.perl -l en | perl moses_scripts/tokenizer.perl -l en -a -q | perl moses_scripts/truecase.perl --model baseline/preprocessed_data/tm.en > baseline/preprocessed_data/tiny_train.en


# rm baseline/preprocessed_data/train.de.p
# rm baseline/preprocessed_data/train.en.p

# without src (DE) and trg (EN) dictionary, `--vocab-src, trg' is empty, must specify `--threshold-src,trg` and  `--num-words-src,trg`
# python preprocess.py --target-lang en --source-lang de --dest-dir baseline/prepared_data_bpe/ --train-prefix baseline/preprocessed_data_bpe/train_bpe --valid-prefix baseline/preprocessed_data_bpe/valid_bpe --test-prefix baseline/preprocessed_data_bpe/test_bpe --tiny-train-prefix baseline/preprocessed_data_bpe/tiny_train_bpe --threshold-src 1 --threshold-tgt 1 --num-words-src 4000 --num-words-tgt 4000

# with src (DE) and trg (EN) dictionary path `--vocab-src,trg`, no need for `--threshold-src,trg` and  `--num-words-src,trg` flags
python preprocess.py --target-lang en --source-lang de --dest-dir baseline/prepared_data_bpe/ --train-prefix baseline/preprocessed_data_bpe/train_bpe --valid-prefix baseline/preprocessed_data_bpe/valid_bpe --test-prefix baseline/preprocessed_data_bpe/test_bpe --tiny-train-prefix baseline/preprocessed_data_bpe/tiny_train_bpe --vocab-src baseline/preprocessed_data_bpe/dict.de --vocab-trg baseline/preprocessed_data_bpe/dict.en
