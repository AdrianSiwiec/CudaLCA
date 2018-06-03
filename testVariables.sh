testsDir=$(realpath ~/storage/tests)
resultTimesDir=resultTimes

defaultBatchSize=-1

singleRunTimeout=300

validityTestsDir=$testsDir/validity
validityAnswersDir=$testsDir/validityOut
validityTestsSizes=(
    10
    1000
    2000
    3000
    4000
    5000
    6000
    7000
    8000
    9000
    1000000
    2000000
)
validityGraspSize=-1
validityOutGenerator="cpuRmqLCA"
validitySolutionToTest="cudaInlabelLCA"

runE1=true
# runE1=false
E1PurgeExistingTests=false
# E1PurgeExistingTests=true
E1SolutionsToTest=(
    "cudaInlabelLCA"
    # "cudaSimpleLCA"
    # "cpuRmqLCA"
)
E1TestsDir=$testsDir/E1
E1ResultsDir=$resultTimesDir/E1
E1TestSizes=(
    1000000
    2000000
    3000000
    5000000
    10000000
    15000000
    20000000
    25000000
    30000000
    40000000
    50000000
    60000000
    70000000
    # 80000000
    # 90000000
    # 100000000
)
E1GraspSizes=( #how far up a father can be
    -1
    10
)
E1DifferentSeeds=3

# runE2=true

runE2=true
E2PurgeExistingTests=true
# E2PurgeExistingTests=false
E2SolutionsToTest=(
    "cudaInlabelLCA"
    "cpuRmqLCA"
)
E2TestsDir=$testsDir/E2
E2ResultsDir=$resultTimesDir/E2
E2TestSizes=(
    10000000
)
E2GraspSizes=(
    -1
)
E2DifferentSeeds=5
E2BatchSizes=(
    1
    10
    100
    1000
    10000
    100000
    1000000
    10000000
)

runE3=true
# runE3=true
E3PurgeExistingTests=true
E3SolutionsToTest=(
    "cudaInlabelLCA"
    "cudaSimpleLCA"
)
E3TestsDir=$testsDir/E3
E3ResultsDir=$resultTimesDir/E3
E3TestSizes=(
    1000000
)
E3GraspSizes=(
    1
    10
    100
    1000
    10000
    100000
    1000000
)
E3DifferentSeeds=5



generateAnswers=true


repeatSingleTest=10

progressBarWidth=50
