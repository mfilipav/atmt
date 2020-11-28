infile=$1
outfile=$2
lang=$3

cat $infile | sed -E 's/(@@ )|(@@ ?$)//g' | perl moses_scripts/detruecase.perl | perl moses_scripts/detokenizer.perl -q -l $lang > $outfile
