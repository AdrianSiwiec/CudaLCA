# solutionForAnswers=cpuRmqLCA.e
solutionForAnswers=cpuSimpleLCA.e

make $solutionForAnswers

for i in $(ls tests); do
    name=tests/${i::-3}.out
    if [[ $i == *.t.in ]]; then
        if [ ! -f $name ]; then
            echo "Generating $name"
            ./$solutionForAnswers <tests/$i >$name
        fi
    fi

    if [[ $i == *.b.in ]]; then
        if [ ! -f $name ]; then
            echo "Generating $name"
            ./$solutionForAnswers tests/$i $name
        fi
    fi
    echo "$name generated."
    echo

done
echo "OK"