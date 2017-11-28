source set_env.sh

DOMAIN=$1
EXP=$2

DATE=`date +"%m_%d_%y"`
TIME=`date +"%H_%M_%S"`
LOGDIR="logs/$DATE"
mkdir -p $LOGDIR
LOGFILE="$LOGDIR/run_log_${DOMAIN}_${EXP}_${TIME}.log"
echo "Saving log to '$LOGFILE'"

REPORTS_DIR="reports/$DATE/${DOMAIN}_${EXP}/"
mkdir -p $REPORTS_DIR
echo "Saving reports to '$REPORTS_DIR'"
echo ""
echo "Note: If you are not starting at stage 0, confirm database exists already."

# Run tests
for MAX_TRAIN in 100 10000
do
    echo ""
    echo "<TEST: Running with following params:>"
    echo "max_train = $MAX_TRAIN" 
    echo ""
    python -u snorkel/contrib/babble/pipelines/run.py \
        --domain $DOMAIN \
        --reports_dir $REPORTS_DIR \
        --start_at 8 \
        --end_at 10 \
        --supervision traditional \
        --max_train $MAX_TRAIN \
	--gen_model_search_space 1 \
        --disc_model_search_space 3 \
	--disc_params_range:batch_size 64 \
	--disc_params_range:n_epochs 30 \
	--disc_params_range:lr 0.001 \
	--disc_params_range:rebalance 0.25 \
        --seed 123 --verbose --no_plots |& tee -a $LOGFILE
done
