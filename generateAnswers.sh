make cpuSimpleLCA.e

for i in $(ls tests); do
    name=tests/${i::-3}.out
    if [[ $i == *.t.in ]]; then
        if [ ! -f $name ]; then
            echo "Generating $name"
            ./cpuSimpleLCA.e <tests/$i >$name
        fi
    fi

    if [[ $i == *.b.in ]]; then
        if [ ! -f $name ]; then
            echo "Generating $name"
            ./cpuSimpleLCA.e tests/$i $name
        fi
    fi
    echo "$name generated."
    echo

done
echo "OK"