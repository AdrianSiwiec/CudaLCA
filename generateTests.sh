function genText {
    name=tests/$1$(echo $2 | numfmt --to=si).$(echo $3 | numfmt --to=si).t.in
    if [ ! -f $name ]; then
        echo "generating $name" 
        ./generate$1.e $2 $3 >$name
    fi
    echo "$name generated" 
    echo
}

function genBin {
    name=tests/$1$(echo $2 | numfmt --to=si).$(echo $3 | numfmt --to=si).b.in
    if [ ! -f $name ]; then
        echo "generating $name" 
        ./generate$1.e $2 $3 $name
    fi
    echo "$name generated" 
    echo
}

mkdir tests
make generateSimple.e
make generateLongSimple.e

# for i in $(seq 1 100);
# do
#   genText Simple 100 $i
#   genBin Simple 100 $i
# done

genText Simple 10 10
genText Simple 9 9
genText Simple 8 8
genText Simple 7 7
genText Simple 10000 10000
genText Simple 100000 100000

genBin Simple 10 10
genBin Simple 1000 1000
genBin Simple 1000000 1000000
genBin Simple 2000000 2000000
genBin Simple 3000000 3000000
# genBin Simple 10000000 10000000

# genBin LongSimple 10000 10000
# genBin LongSimple 100000 100000
# genBin LongSimple 1000000 1000000

echo "OK"
