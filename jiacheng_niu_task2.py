from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, Rating
import sys
import re
import time
import math
import random

def rmse(predictions, evalData):
    ratesAndPreds = evalData.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
    RMSE = math.sqrt(ratesAndPreds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean())
    print('------------\nRoot Mean Squared Error = ' + str(RMSE) + '\n------------\n')

def modelBasedCF():
    f = open(train_file, 'r')
    lines = f.readlines()[1:]
    users, businesses = {}, {}
    for i in range(len(lines)):
        lines[i] = lines[i].split(',')
        users[lines[i][0]] = 1
        businesses[lines[i][1]] = 1

    users = list(users.keys())
    businesses = list(businesses.keys())
    uMap, bMap = {users[i]: i for i in range(len(users))}, {businesses[i]: i for i in range(len(businesses))}

    ratings = sc.parallelize(lines).map(lambda l: Rating(uMap[l[0]], bMap[l[1]], float(l[2])))

    model = ALS.train(ratings, rank=5, iterations=5, lambda_=0.1, nonnegative=True)

    evalData = sc.textFile(test_file).map(lambda l: l.split(',')) \
        .filter(lambda l: len(l) > 2 and len(re.findall('[0-9]', l[2])) > 0).map(lambda l: (l[0], l[1], float(l[2])))
    missingData = evalData.filter(lambda l: l[0] not in uMap or l[1] not in bMap)
    testData = evalData.subtract(missingData).map(lambda l: (uMap[l[0]], bMap[l[1]]))
    predictions = model.predictAll(testData).map(lambda r: ((users[r[0]], businesses[r[1]]), r[2]))\
        .union(missingData.map(lambda l: ((l[0], l[1]), 3.7)))

    res = predictions.collect()
    writeFile(res)
    # rmse(predictions, evalData)


def userBasedCF():
    def mapper(tup):
        u, b = tup
        cands = b_u[b]
        weights = {}
        for cand in cands:
            commons = u_b[u]['__all__'] & u_b[cand]['__all__']
            num = len(commons)
            if num < 2:
                continue
            flag_u, flag_cand = False, False
            u_avg = sum([u_b[u][c] for c in commons]) / num
            cand_avg = sum([u_b[cand][c] for c in commons]) / num
            for c in commons:
                flag_u = flag_u or u_b[u][c] != u_avg
                flag_cand = flag_cand or u_b[cand][c] != cand_avg
                if (flag_u and flag_cand):
                    break
            if (flag_u and flag_cand) is False:
                continue
            weights[cand] = sum([(u_b[u][c] - u_avg) * (u_b[cand][c] - cand_avg) for c in commons]) \
                            / math.sqrt(sum([(u_b[u][c] - u_avg) ** 2 for c in commons])) \
                            / math.sqrt(sum([(u_b[cand][c] - cand_avg) ** 2 for c in commons]))

        score = u_b[u]['__sum__'] / u_b[u]['__num__']
        cands_weights = sorted(weights.items(), key=lambda a: a[1], reverse=True)
        cands_weights = cands_weights[: min(len(cands_weights), 10)]
        sum_weight = 0
        sum_score = 0
        for pair in cands_weights:
            sum_score += (u_b[pair[0]][b] - u_b[pair[0]]['__sum__'] / u_b[pair[0]]['__num__']) * pair[1]
            sum_weight += abs(pair[1])
        if sum_weight > 0:
            score += sum_score / sum_weight
        score = min(max(1.0, score), 5.0)
        return ((u, b), score)

    f = open(train_file, 'r')
    lines = f.readlines()[1:]
    u_b = {}
    b_u = {}
    for line in lines:
        u, b, score = line.split(',')
        score = float(score)
        if u not in u_b:
            u_b[u] = {'__all__': set([b]), '__sum__': score, '__num__': 1}
        else:
            u_b[u]['__all__'].add(b)
            u_b[u]['__sum__'] += score
            u_b[u]['__num__'] += 1
        u_b[u][b] = score
        if b not in b_u:
            b_u[b] = set([])
        b_u[b].add(u)
    evalData = sc.textFile(test_file).map(lambda l: l.split(',')) \
        .filter(lambda l: len(l) > 2 and len(re.findall('[0-9]', l[2])) > 0).map(lambda l: (l[0], l[1], float(l[2])))
    missingData = evalData.filter(lambda l: l[0] not in u_b or l[1] not in b_u)
    testData = evalData.subtract(missingData).repartition(64).map(lambda l: (l[0], l[1]))
    predictions = testData.map(mapper).union(missingData.map(lambda l: ((l[0], l[1]), 3.7)))

    res = predictions.collect()
    writeFile(res)
    # rmse(predictions, evalData)


def itemBasedCF():
    def mapper(tup):
        u, b = tup
        cands = u_b[u]
        weights = {}
        for cand in cands:
            commons = b_u[b]['__all__'] & b_u[cand]['__all__']
            num = len(commons)
            if num < 2:
                continue
            flag_b, flag_cand = False, False
            b_avg = sum([b_u[b][c] for c in commons]) / num
            cand_avg = sum([b_u[cand][c] for c in commons]) / num
            for c in commons:
                flag_b = flag_b or b_u[b][c] != b_avg
                flag_cand = flag_cand or b_u[cand][c] != cand_avg
                if (flag_b and flag_cand):
                    break
            if (flag_b and flag_cand) is False:
                continue
            weights[cand] = sum([(b_u[b][c] - b_avg) * (b_u[cand][c] - cand_avg) for c in commons]) \
                            / math.sqrt(sum([(b_u[b][c] - b_avg) ** 2 for c in commons])) \
                            / math.sqrt(sum([(b_u[cand][c] - cand_avg) ** 2 for c in commons]))

        score = b_u[b]['__sum__'] / b_u[b]['__num__']
        cands_weights = sorted(weights.items(), key=lambda a: a[1], reverse=True)
        cands_weights = cands_weights[: min(len(cands_weights), 10)]
        sum_weight = 0
        sum_score = 0
        for pair in cands_weights:
            sum_score += (b_u[pair[0]][u] - b_u[pair[0]]['__sum__'] / b_u[pair[0]]['__num__']) * pair[1]
            sum_weight += abs(pair[1])
        if sum_weight > 0:
            score += sum_score / sum_weight
        score = min(max(1.0, score), 5.0)
        return ((u, b), score)

    f = open(train_file, 'r')
    lines = f.readlines()[1:]
    u_b = {}
    b_u = {}
    for line in lines:
        u, b, score = line.split(',')
        score = float(score)
        if b not in b_u:
            b_u[b] = {'__all__': set([u]), '__sum__': score, '__num__': 1}
        else:
            b_u[b]['__all__'].add(u)
            b_u[b]['__sum__'] += score
            b_u[b]['__num__'] += 1
        b_u[b][u] = score
        if u not in u_b:
            u_b[u] = set([])
        u_b[u].add(b)
    evalData = sc.textFile(test_file).map(lambda l: l.split(',')) \
        .filter(lambda l: len(l) > 2 and len(re.findall('[0-9]', l[2])) > 0).map(lambda l: (l[0], l[1], float(l[2])))
    missingData = evalData.filter(lambda l: l[0] not in u_b or l[1] not in b_u)
    testData = evalData.subtract(missingData).repartition(64).map(lambda l: (l[0], l[1]))
    predictions = testData.map(mapper).union(missingData.map(lambda l: ((l[0], l[1]), 3.7)))

    res = predictions.collect()
    writeFile(res)
    # rmse(predictions, evalData)


def itemBasedCF_LSH():
    def mapper(tup):
        u, b = tup
        cands = u_b[u]
        weights = {}
        for cand in cands:
            commons = b_u[b]['__all__'] & b_u[cand]['__all__']
            if b in b_b:
                commons = list(filter(lambda x: x in b_b[b], list(commons)))
            num = len(commons)
            if num < 2:
                continue
            flag_b, flag_cand = False, False
            b_avg = sum([b_u[b][c] for c in commons]) / num
            cand_avg = sum([b_u[cand][c] for c in commons]) / num
            for c in commons:
                flag_b = flag_b or b_u[b][c] != b_avg
                flag_cand = flag_cand or b_u[cand][c] != cand_avg
                if (flag_b and flag_cand):
                    break
            if (flag_b and flag_cand) is False:
                continue
            weights[cand] = sum([(b_u[b][c] - b_avg) * (b_u[cand][c] - cand_avg) for c in commons]) \
                            / math.sqrt(sum([(b_u[b][c] - b_avg) ** 2 for c in commons])) \
                            / math.sqrt(sum([(b_u[cand][c] - cand_avg) ** 2 for c in commons]))
            if b in b_b:
                weights[cand] *= b_b[b][cand]

        score = b_u[b]['__sum__'] / b_u[b]['__num__']
        cands_weights = sorted(weights.items(), key=lambda a: a[1], reverse=True)
        cands_weights = cands_weights[: min(len(cands_weights), 10)]
        sum_weight = 0
        sum_score = 0
        for pair in cands_weights:
            sum_score += (b_u[pair[0]][u] - b_u[pair[0]]['__sum__'] / b_u[pair[0]]['__num__']) * pair[1]
            sum_weight += abs(pair[1])
        if sum_weight > 0:
            score += sum_score / sum_weight
        score = min(max(1.0, score), 5.0)
        return ((u, b), score)

    def LSH(users, businesses, b_u):
        def minHashHelper():
            def getNPrime(n):
                primes = []
                dp = [True] * (n + 1)
                for i in range(7, n + 1):
                    if dp[i]:
                        flag = True
                        for j in range(2, int(math.sqrt(i)) + 1):
                            if i % j == 0:
                                flag = False
                                break
                        if flag:
                            primes.append(i)
                            for k in range(2, int(n / i) + 1):
                                dp[i * k] = False
                return primes

            a_list = []
            b_list = []
            primes = getNPrime(2000)
            length = len(primes)
            for _ in range(SIZE):
                a_list.append(primes[random.randint(0, length - 1)])
                b_list.append(primes[random.randint(0, length - 1)])
            hash_table = []
            for i in range(SIZE):
                hash_table.append({})
                for j in range(n_user):
                    key = (a_list[i] * j + b_list[i]) % n_user
                    hash_table[i][j] = key
            return hash_table

        def minHashMapper(b):
            u_ids = [users[u] for u in b_u[b]['__all__']]
            res = [None] * SIZE
            for i in range(SIZE):
                for u_id in u_ids:
                    if res[i] == None or res[i] > hash_table[i][u_id]:
                        res[i] = hash_table[i][u_id]

            return (b, res)

        def LSHMapper(tup):
            id, sign = tup
            sign = list(map(str, sign))
            res = []
            for i in range(B):
                res.append((tuple([i] + sign[i * R: (i + 1) * R]), id))
            return res

        def candMapper(cands):
            cands = sorted(cands)
            res = []
            for i in range(len(cands) - 1):
                for j in range(i + 1, len(cands)):
                    res.append((cands[i], cands[j]))
            return res

        def jacardMapper(pair):
            a, b = b_u[pair[0]]['__all__'], b_u[pair[1]]['__all__']
            num, total = len(a & b), len(a | b)
            score = float(num) / total
            if score >= 0.1:
                return (pair, score)

        SIZE = 100
        B = 50
        R = int(SIZE / B)
        u_list = sorted(users.keys())
        b_list = sorted(businesses.keys())
        users = {u_list[i]: i for i in range(len(u_list))}
        n_user = len(u_list)
        hash_table = minHashHelper()
        rdd = sc.parallelize(b_list, 16).map(minHashMapper)
        # LSH
        rdd = rdd.flatMap(LSHMapper).groupByKey().map(lambda x: x[1]).filter(lambda x: len(x) > 1)
        # candidates
        rdd = rdd.flatMap(candMapper).distinct()
        # jacard
        rdd = rdd.map(jacardMapper).filter(lambda x: x != None)
        # final
        res = rdd.collect()
        b_b = {}
        for tup in res:
            if tup[0][0] not in b_b:
                b_b[tup[0][0]] = {}
            if tup[0][1] not in b_b:
                b_b[tup[0][1]] = {}
            b_b[tup[0][0]][tup[0][1]] = tup[1]
            b_b[tup[0][1]][tup[0][0]] = tup[1]
        return b_b


    f = open(train_file, 'r')
    lines = f.readlines()[1:]
    u_b = {}
    b_u = {}
    users, businesses = {}, {}
    for line in lines:
        u, b, score = line.split(',')
        users[u] = 1
        businesses[b] = []
        score = float(score)
        if b not in b_u:
            b_u[b] = {'__all__': set([u]), '__sum__': score, '__num__': 1}
        else:
            b_u[b]['__all__'].add(u)
            b_u[b]['__sum__'] += score
            b_u[b]['__num__'] += 1
        b_u[b][u] = score
        if u not in u_b:
            u_b[u] = set([])
        u_b[u].add(b)
    b_b = LSH(users, businesses, b_u)

    evalData = sc.textFile(test_file).map(lambda l: l.split(',')) \
        .filter(lambda l: len(l) > 2 and len(re.findall('[0-9]', l[2])) > 0).map(lambda l: (l[0], l[1], float(l[2])))
    missingData = evalData.filter(lambda l: l[0] not in u_b or l[1] not in b_u)
    testData = evalData.subtract(missingData).repartition(64).map(lambda l: (l[0], l[1]))
    predictions = testData.map(mapper).union(missingData.map(lambda l: ((l[0], l[1]), 3.7)))

    res = predictions.collect()
    writeFile(res)
    # rmse(predictions, evalData)


def writeFile(res):
    f = open(output_file, 'w')
    f.write('user_id,business_id,prediction\n')
    for tup in res:
        f.write(tup[0][0] + ',' + tup[0][1] + ',' + str(tup[1]) + '\n')
    f.close()


if __name__ == '__main__':
    start = time.time()
    sc = SparkContext(appName="inf553")
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    case = int(sys.argv[3])
    output_file = sys.argv[4]
    if case == 1:
        modelBasedCF()
    elif case == 2:
        userBasedCF()
    elif case == 3:
        itemBasedCF()
    else:
        itemBasedCF_LSH()
    end = time.time()
    print('Duration:', end - start)