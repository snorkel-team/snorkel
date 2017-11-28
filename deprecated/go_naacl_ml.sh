source set_env.sh

DOMAIN=$1
EXP=$2

DATE=`date +"%m_%d_%y"`
TIME=`date +"%H_%M_%S"`
LOGDIR="logs/$DATE"
mkdir -p $LOGDIR

REPORTS_DIR="reports/$DATE/${DOMAIN}_${EXP}/"
mkdir -p $REPORTS_DIR
echo "Saving reports to '$REPORTS_DIR'"
echo ""
echo "Note: If you are not starting at stage 0, confirm database exists already."

# Run tests
for MAX_TRAIN in 0 30 60 90 150 300 1500 3000 6667
do
    echo ""
    echo "<TEST: Running with following params:>"
    echo "max_train = $MAX_TRAIN" 
    echo ""
    DB_NAME="babble_${DOMAIN}_${EXP}_${MAX_TRAIN}"
    echo "babble_${DOMAIN}_${EXP}_${MAX_TRAIN}"
    echo "Using db: $DB_NAME"
    cp babble_${DOMAIN}_labeled_tocopy.db $DB_NAME.db
    LOGFILE="$LOGDIR/run_log_${DB_NAME}_${TIME}.log"
    echo "Saving log to '$LOGFILE'"
    python -u snorkel/contrib/babble/pipelines/run.py \
        --domain $DOMAIN \
	--db_name $DB_NAME \
        --reports_dir $REPORTS_DIR \
        --start_at 8 \
        --supervision traditional \
        --max_train $MAX_TRAIN \
	--disc_model_class lstm \
        --disc_model_search_space 10 \
        --verbose --no_plots |& tee -a $LOGFILE
done
