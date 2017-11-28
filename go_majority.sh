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

for ITER in 1 2 3
do

RUN="${DOMAIN}_${EXP}_${TIME}_${ITER}"

DB_NAME="babble_${RUN}"
echo "Using db: $DB_NAME"
cp babble_${DOMAIN}_labeled_tocopy.db $DB_NAME.db

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
    --supervision majority \
    --disc_model_class lstm \
    --disc_model_search_space 10 \
    --verbose --no_plots |& tee -a $LOGFILE &
sleep 3
done
