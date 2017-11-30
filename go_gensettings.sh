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

for learn_deps in True False
do
for lf_propensity in True False 
do
for lf_class_propensity in True False 
do
for class_prior in True False 
do


RUN="${DOMAIN}_${EXP}_${TIME}_${learn_deps}_${lf_propensity}_${lf_class_propensity}_${class_prior}_${ITER}"

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
    --start_at 7 \
    --end_at 10 \
    --supervision generative \
    --learn_deps $learn_deps \
    --gen_init_params:lf_propensity $lf_propensity \
    --gen_init_params:lf_class_propensity $lf_class_propensity \
    --gen_init_params:class_prior $class_prior \
    --gen_model_search_space 20 \
    --disc_model_class lstm \
    --disc_model_search_space 10 \
    --verbose --no_plots |& tee -a $LOGFILE &
sleep 3

done
done
done
done
done
