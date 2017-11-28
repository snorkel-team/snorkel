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
for ITER in 1 2 3
do
for MAX_TRAIN in 30 60 90 150 300 600 900 1500 3000 10000
do
    echo ""
    echo "<TEST: Running with following params:>"
    echo "max_train = $MAX_TRAIN" 
    echo ""
    LOGFILE="$LOGDIR/run_log_${DOMAIN}_${EXP}_${TIME}_${MAX_TRAIN}_${ITER}.log"
    echo "Saving log to '$LOGFILE'"
    echo ""
    python -u snorkel/contrib/babble/pipelines/run.py \
        --domain $DOMAIN \
        --reports_dir $REPORTS_DIR \
        --start_at 7 \
        --supervision traditional \
        --max_train $MAX_TRAIN \
	--disc_model_class lstm \
        --disc_model_search_space 10 \
        --verbose --no_plots |& tee -a $LOGFILE
done
done
