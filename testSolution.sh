# toTest="cudaSimpleLCA"
toTest="cudaInlabelLCA"
# toTest="cpuRmqLCA"

# echo "Generating Tests"
# ./generateTests.sh 
echo "Generating Answers"
./generateAnswers.sh 

# testsDir=$(realpath ~/storage/tests)
testsDir=tests

make $toTest.e

for i in $(ls $testsDir/*.b.in); do
    i=$(basename $i)
    outName=$testsDir/${i::-3}.out
    echo "Testing on $i"
    ./$toTest.e $testsDir/$i out 2>/dev/null
    if diff out $outName >/dev/null; then
        echo "$i OK"
    else
        echo "Wrong answer on $i. Aborting"
        exit 1
    fi
done

rm out
echo "All OK"
