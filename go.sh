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

echo ""
echo "<TEST:>"
echo ""
python -u snorkel/contrib/babble/pipelines/run.py \
    --domain $DOMAIN \
    --reports_dir $REPORTS_DIR \
    --start_at 7 \
    --end_at 10 \
    --supervision traditional \
    --gen_model_search_space 1 \
    --disc_model_search_space 1 \
    --disc_params_search:max_steps 1 \
    --seed 1023 --verbose --no_plots |& tee -a $LOGFILE
