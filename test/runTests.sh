TESTS_PATH=$SNORKELHOME/test

if [ $# -eq 0 ]
    then
	    echo "Running all tests"
	    for test in $TESTS_PATH/*Tests.py
	    do
	            echo "Test: $test"
	            python $test
	    done
else
    test=$1
    echo "Running Test: $test"
    python $TESTS_PATH/$test
fi
