source set_env.sh

DOMAIN=$1

DATE=`date +"%m_%d_%y"`
TIME=`date +"%H_%M_%S"`
LOGDIR="logs/$DATE"
mkdir -p $LOGDIR
LOGFILE="$LOGDIR/run_log_${DOMAIN}_${TIME}.log"
echo "Saving log to '$LOGFILE'"

REPORTS_DIR="reports/gen_tests_$DOMAIN/"
mkdir -p $REPORTS_DIR
echo "Saving reports to '$REPORTS_DIR'"
echo ""
echo "Note: Consider if pipeline up through stage 5 must have already been run."

for class_prior in True False
do
    echo ""
    echo "<TEST: Running for class_prior = $class_prior>"
    python -u snorkel/contrib/babble/pipelines/run.py \
        --domain $DOMAIN \
        --reports_dir $REPORTS_DIR \
        --class_prior $class_prior \
        --postgres \
        --start_at 6 \
        --debug \
        --verbose --no_plots | tee -a $LOGFILE
        # --debug \
        # --end_at 6 \
done
