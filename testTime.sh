# toTest="cudaInlabelLCA"
toTest="cudaSimpleLCA"
# toTest="cpuRmqLCA"

echo "Generating Tests"
./generateTests.sh
# echo "Generating Answers"
# ./generateAnswers.sh 

make $toTest.e

mkdir timesResults

repeatTimes=20

progressBarWidth=30

for test in $(ls tests/*.b.in); do
    outName=timesResults/${toTest}\#${test:6:-5}\#$(date '+%Y.%m.%d.%H.%M.%S').out
    touch $outName

    echo -n "Running ${test:6:-5}"
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

        ./$toTest.e $test /dev/null 2>>$outName
        echo "" >>$outName
    done
    echo ""
done