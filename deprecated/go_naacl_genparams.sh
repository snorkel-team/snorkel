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
for learn_deps in True False
do
for lf_propensity in True False 
do
for lf_class_propensity in True False 
do
for class_prior in True False 
do
    echo ""
    echo "<TEST: Running with following params:>"
    echo "learn_deps = $learn_deps"
    echo "lf_propensity = $lf_propensity" 
    echo "lf_class_propensity = $lf_class_propensity"
    echo "class_prior = $class_prior"
    echo ""
    LOGFILE="$LOGDIR/run_log_${DOMAIN}_${EXP}_${TIME}_${learn_deps}_${lf_propensity}_${lf_class_propensity}_${class_prior}.log"
    echo "Saving log to '$LOGFILE'"
    DBNAME=babble_${DOMAIN}_${EXP}_${learn_deps}_${lf_propensity}_${lf_class_propensity}_${class_prior}
    echo "Using dbname '$DBNAME'"
    cp babble_${DOMAIN}_labeled_tocopy.db ${DBNAME}.db
    python -u snorkel/contrib/babble/pipelines/run.py \
        --domain $DOMAIN \
	--db_name $DBNAME \
        --reports_dir $REPORTS_DIR \
	--supervision generative \
	--learn_deps $learn_deps \
        --gen_init_params:lf_propensity $lf_propensity \
        --gen_init_params:lf_class_propensity $lf_class_propensity \
        --gen_init_params:class_prior $class_prior \
        --start_at 7 \
        --end_at 10 \
        --gen_model_search_space 15 \
	--disc_model_search_space 15 \
        --verbose --no_plots |& tee -a $LOGFILE &
    sleep 1
done
done
done
done
