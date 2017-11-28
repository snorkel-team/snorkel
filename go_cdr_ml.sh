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

for ITER in 1
do

for MAX_TRAIN in 30 60 90 150 300 1500 3000 6667
do

RUN="${DOMAIN}_${EXP}_${TIME}_${MAX_TRAIN}_${ITER}"

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
    --supervision traditional \
    --gen_model_search_space 1 \
    --max_train $MAX_TRAIN \
    --disc_model_class lstm \
    --disc_model_search_space 10 \
    --verbose --no_plots |& tee -a $LOGFILE &
sleep 3

done
done
