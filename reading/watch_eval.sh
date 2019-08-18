MODEL_DIR=$1
export DATA_PATH=$2
inotifywait -m $MODEL_DIR -e create -e moved_to |
    while read path action file; do
        if [[ "$file" =~ .*pt$ ]]; then # Does the file end with .xml?
            export RESUME_FROM="$MODEL_DIR/$file" # If so, do your thing here!
            export ITER=$(basename $file .pt | cut -d'_' -f 3)
            export DF_PATH="$RESUME_FROM.csv"
            python reading/eval_reader.py
        fi
    done