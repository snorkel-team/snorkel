DOMAIN=$1
EXP=$2

DATE=`date +"%m_%d_%y"`
TIME=`date +"%H_%M_%S"`

LOGDIR="logs/$DATE"
mkdir -p $LOGDIR

REPORTS_DIR="reports/$DATE"
mkdir -p $REPORTS_DIR

echo ""
echo "<TEST:>"
echo ""

for temp in 1 2
do

RUN="${DOMAIN}_${EXP}_${TIME}_${TEMP}"

REPORTS_SUBDIR="$REPORTS_DIR/$RUN/"
mkdir -p $REPORTS_SUBDIR
echo "Saving reports to '$REPORTS_SUBDIR'"

LOGFILE="$LOGDIR/$RUN.log"
echo "Saving log to '$LOGFILE'"

python -u snorkel/contrib/babble/pipelines/run.py \
    --domain $DOMAIN \
    --reports_dir $REPORTS_SUBDIR \
    --start_at 8 \
    --end_at 10 \
    --supervision traditional \
    --gen_model_search_space 1 \
    --max_train 500 \
    --disc_model_class logreg \
    --disc_model_search_space 2 \
    --disc_params_range:n_epochs 2 \
    --verbose --no_plots |& tee -a $LOGFILE
done