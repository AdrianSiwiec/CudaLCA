#toTest="cudaSimpleLCA"
toTest="cudaInlabelLCA"

echo "Generating Tests"
./generateTests.sh 
echo "Generating Answers"
./generateAnswers.sh 

make $toTest.e

for i in $(ls tests/*.b.in); do
    outName=${i::-3}.out
    ./$toTest.e $i out 2>/dev/null
    echo "Testing on $i"
    if diff out $outName >/dev/null; then
        echo "$i OK"
    else
        echo "Wrong answer on $i. Aborting"
        exit 1
    fi
done

rm out
echo "All OK"
