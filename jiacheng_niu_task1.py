from pyspark import SparkContext
import sys
import time
import math
import random

SIZE = 120
B = 40
R = int(SIZE / B)

def readFile(fileName):
    users, businesses = {}, {}
    business_user = {}
    f = open(fileName, 'r')
    lines = f.readlines()[1:]
    for line in lines:
        u, b, _ = line.split(',')
        users[u] = 1
        businesses[b] = 1
        if b not in business_user:
            business_user[b] = set([u])
        else:
            business_user[b].add(u)
    u_list = sorted(users.keys())
    b_list = sorted(businesses.keys())
    users = {u_list[i]: i for i in range(len(u_list))}

    matrix = [[0] * len(u_list) for _ in range(len(b_list))]
    for b in business_user:
        for u in business_user[b]:
            matrix[businesses[b]][users[u]] = 1
    return business_user, users, b_list


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
    hash_list = []
    for i in range(SIZE):
        hash_list.append({})
        for j in range(n_user):
            key = (a_list[i] * j + b_list[i]) % n_user
            hash_list[i][j] = key
    return hash_list


def minHashMapper(b):
    u_ids = [users[u] for u in business_user[b]]
    res = [None] * SIZE
    for i in range(SIZE):
        for u_id in u_ids:
            if res[i] == None or res[i] > hash_list[i][u_id]:
                res[i] = hash_list[i][u_id]

    return (b, res)

def LSHMapper(tup):
    id, sign = tup
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
    a, b = business_user[pair[0]], business_user[pair[1]]
    num, total = len(a & b), len(a | b)
    score = float(num) / total

    if score >= 0.5:
        return (pair, score)


def writeFile(res):
    f = open(sys.argv[2], 'w')
    f.write('business_id_1,business_id_2,similarity\n')
    for tup in res:
        f.write(tup[0][0] + ',' + tup[0][1] + ',' + str(tup[1]) + '\n')
    f.close()


if __name__ == '__main__':
    start = time.time()
    sc = SparkContext(appName="inf553")
    business_user, users, b_list = readFile(sys.argv[1])
    n_business = len(b_list)
    n_user = len(users)
    hash_list = minHashHelper()
    #min hash
    rdd = sc.parallelize(b_list, 16).map(minHashMapper)
    #LSH
    rdd = rdd.flatMap(LSHMapper).groupByKey().map(lambda x: x[1]).filter(lambda x: len(x) > 1)
    #candidates
    rdd = rdd.flatMap(candMapper).distinct()
    #jacard
    rdd = rdd.map(jacardMapper).filter(lambda x: x != None)
    #final
    res = rdd.sortBy(lambda x: x[0]).collect()
    writeFile(res)
    end = time.time()
    print('Duration:', end - start)