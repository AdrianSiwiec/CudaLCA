function genText {
    ./generate$1.e $2 $3 >tests/$1$(echo $2 | numfmt --to=si).t.in
}

function genBin {
    ./generate$1.e $2 $3 tests/$1$(echo $2 | numfmt --to=si).b.in
}

mkdir tests
make generateSimple.e

genText Simple 10 10
genText Simple 10000 10000
genText Simple 1000000 1000000
genText Simple 10000000 10000000

genBin Simple 10 10
genBin Simple 1000000 1000000
genBin Simple 10000000 10000000

echo "OK"