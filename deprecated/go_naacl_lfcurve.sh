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
for MAX_EXP in 5 10 15 20 25 30 35 40
do
DBNAME=babble_${DOMAIN}_${EXP}_${MAX_EXP}_${ITER}
echo "Using dbname '$DBNAME'"
LOGFILE="$LOGDIR/run_log_${DBNAME}_${TIME}.log"
echo "Saving log to '$LOGFILE'"
cp babble_${DOMAIN}_featurized_tocopy.db ${DBNAME}.db
echo ""
echo "<TEST: Running with following params:>"
echo "max_explanations = $MAX_EXP"
echo ""
python -u snorkel/contrib/babble/pipelines/run.py \
    --domain $DOMAIN \
    --reports_dir $REPORTS_DIR \
    --db_name $DBNAME \
    --start_at 5 \
    --max_explanations $MAX_EXP \
    --supervision generative \
    --gen_init_params:lf_propensity true \
    --gen_init_params:class_prior true \
    --gen_model_search_space 20 \
    --disc_model_search_space 20 \
    --disc_model_class lstm \
    --verbose --no_plots |& tee -a $LOGFILE &
sleep 2
done
done
