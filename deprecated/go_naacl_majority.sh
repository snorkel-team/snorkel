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
echo ""
echo "<TEST: Running with following params:>"
echo ""

for ITER in 1 2 3
do
LOGFILE="$LOGDIR/run_log_${DOMAIN}_${EXP}_${TIME}_majority_${ITER}.log"
echo "Saving log to '$LOGFILE'"
DBNAME=babble_${DOMAIN}_${EXP}_majority_${ITER}
echo "Using dbname '$DBNAME'"
cp babble_${DOMAIN}_labeled_tocopy.db ${DBNAME}.db
python -u snorkel/contrib/babble/pipelines/run.py \
        --domain $DOMAIN \
	--db_name $DBNAME \
        --reports_dir $REPORTS_DIR \
	--supervision majority \
        --disc_model_class lstm \
        --start_at 7 \
        --end_at 10 \
	--disc_model_search_space 20 \
        --verbose --no_plots |& tee -a $LOGFILE &
sleep 3
done
