toTest="cudaSimpleLCA"

./generateTests.sh >/dev/null
./generateAnswers.sh >/dev/null
make $toTest.e

for i in $(ls tests/*.b.in); do
    outName=${i::-3}.out
    ./$toTest.e $i out 2>/dev/null
    if diff out $outName >/dev/null; then
        echo "$i OK"
    else
        echo "Wrong answer on $i. Aborting"
        exit 1
    fi
done

rm out
echo "All OK"