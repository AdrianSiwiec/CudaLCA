solutionsToTest=(
    "cudaInlabelLCA"
    # "cudaSimpleLCA"
    # "cpuRmqLCA"
)


# echo "Generating Tests"
# ./generateTests.sh
# echo "Generating Answers"
# ./generateAnswers.sh 

# testsDir=$(realpath ~/storage/tests)
testsDir=tests

mkdir timesResults

repeatTimes=10

progressBarWidth=30

for toTest in ${solutionsToTest[@]}; do
    make $toTest.e

    echo $toTest
    for test in $testsDir/*.b.in; do
        test=$(basename $test)
        outName=timesResults/${toTest}\#${test::-5}\#$(date '+%Y.%m.%d.%H.%M.%S' -d @$(stat -c %Y $toTest.e)).out
        touch $outName

        echo -n "Running ${test}"
        echo ""

        for i in $(seq 1 $repeatTimes); do
            #progress bar
            progress=$(($i*$progressBarWidth/$repeatTimes))
            bar="|"
            for k in $(seq 1 $progress); do
                bar=$bar"#"
            done
            for k in $(seq 1 $(($progressBarWidth-$progress))); do
                bar=$bar"-"
            done
            bar=$bar"|"
            echo -ne "$bar\r"
            #progress bar end
    
            ./$toTest.e $testsDir/$test /dev/null 2>>$outName
            echo "" >>$outName
        done
        echo ""
    done
done