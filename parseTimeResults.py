#!/usr/bin/python3
import os
import sys

resultsDirectory = ""
if len(sys.argv) <= 1:
    resultsDirectory = "./timesResults/"
else:
    resultsDirectory = sys.argv[1]


class result:
    def __init__(self, solutionName, testName, timestamp, combined, singleResults):
        # self.solutionName = solutionName
        self.solutionName = solutionName+" ("+timestamp+")"
        self.testName = testName
        self.timestamp = timestamp
        self.combined = combined
        self.singleResults = singleResults

        numOfQueries = 0
        combinedQueryTimes = 0.0
        for s in singleResults:
            numOfQueries += s.numOfQueries
            combinedQueryTimes += s.combinedQueryTimes

        if(numOfQueries > 0):
            self.oneQueryTime = combinedQueryTimes/numOfQueries
            self.combined.append(("AvgQueryTime(ns)", self.oneQueryTime*10**9))
        else:
            self.oneQueryTime = 0

    # def __repr__(self):
    #     return "Solution: "+self.solutionName+", on: "+self.testName+". Timestamp: "+self.timestamp+"\n"+str(self.combined)


class singleResult:  # whole
    def __init__(self, rawSingleResult):
        self.sectionNames = []
        self.sectionTimes = []

        self.numOfQueries = 0
        self.combinedQueryTimes = 0.0

        res = rawSingleResult.split("\n")
        for line in res:
            if "Whole" in line:
                sectionName, time, tmp = line.split(",")
                self.sectionNames.append(sectionName)
                self.sectionTimes.append(float(time))
                if "Queries" in line:
                    self.combinedQueryTimes += float(time)
            if "NumOfQueries" in line:
                sectionName, time, q = line.split(",")
                self.numOfQueries += int(q.split(":")[1])


def getResultFromFilename(filename):
    solutionName, testName, timestamp = filename[:-4].split("#")

    content = []
    with open(resultsDirectory+filename) as rawResult:
        content.extend(rawResult.read().split("\n\n"))

    singleResults = []

    for res in content:
        if len(res) > 1:
            singleResults.append(singleResult(res))

    results = []

    names = singleResults[0].sectionNames
    times = []

    for res in singleResults:
        for i in range(len(res.sectionNames)):
            if len(times) <= i:
                times.append([])
            times[i].append(res.sectionTimes[i])

    for i in range(len(names)):
        time = 0.0
        for t in times[i]:
            time += t

        time /= len(times[i])
        results.append((names[i], time))

    return result(solutionName, testName, timestamp, results, singleResults)


resultsRaw = []
for(dirpath, dirnames, filenames) in os.walk(resultsDirectory):
    resultsRaw.extend(filenames)
    break

results = []
for filename in resultsRaw:
    if filename[0] != '.':
        results.append(getResultFromFilename(filename))

testNames = []
for res in results:
    if not res.testName in testNames:
        testNames.append(res.testName)

solutions = []
for res in results:
    if not res.solutionName in solutions:
        solutions.append(res.solutionName)


for testName in testNames:
    print("Results on "+testName)

    for solution in solutions:
        print(" "+solution+":")

        for res in results:
            if res.solutionName == solution and res.testName == testName:
                print("   (timestamp: "+res.timestamp)
                for c in res.combined:
                    print('   {:<18}{:>18}ms'.format(c[0], str(round(c[1],4))))

                print('   {:<18}{:>18}ns'.format("AverageQueryTime", round(res.oneQueryTime*10**9, 4)))
                print()
    print()

testPrefixes = ["Simple", "LongSimple", "kron_g500", "road"]
sectionNames = ["Preprocessing", "AvgQueryTime(ns)", "List Rank"]

for testPrefix in testPrefixes:
    print(testPrefix)
    for sectionName in sectionNames:
        print(sectionName+",", end="")
        for testName in testNames:
            if testName.startswith(testPrefix):
                print(testName + ",", end="")

        print()

        for solution in solutions:
            print(solution+",", end="")
            for res in results:
                for testName in testNames:
                    if  testName.startswith(testPrefix) and res.solutionName == solution and res.testName == testName:
                        for sectionRes in res.combined:
                            if(sectionRes[0] == sectionName):
                                print(str(round(sectionRes[1], 4))+",", end="")
            print()

        print()

    print()


# print(results)
