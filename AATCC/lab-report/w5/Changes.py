
import time
k = 0

def recMC(coinValueList,change):
    global k
    k = k + 1
    # print(k)

    minCoins = change
    if change in coinValueList:
        return 1
    else:
        for i in [c for c in coinValueList if c <= change]:
            numCoins = 1 + recMC(coinValueList, change-i)
            if numCoins < minCoins:
                minCoins = numCoins
    return minCoins


start = time.time()
# AA = recMC([1,5,10,25], 11)
AA = recMC([1, 5, 10, 25], 26)
# AA = recMC([1, 5, 10, 25], 63)
end = time.time()
print(AA)
print("time is %.4f" % (float(end)-float(start)))
print(k)




k = 0

def recDC(coinValueList, change, knownResults):
    global k
    # k = k + 1
    # print(k)
    minCoins = change
    if change in coinValueList:
        knownResults[change] = 1
        return 1
    elif knownResults[change] > 0:
        return knownResults[change]
    else:
        for i in [c for c in coinValueList if c <= change]:
            numCoins = 1 + recDC(coinValueList, change-i, knownResults)
            if numCoins < minCoins:
                minCoins = numCoins
                knownResults[change] = minCoins
        k = k + 1
    return minCoins

start1 = time.time()
# AA = recDC([1, 5, 10, 25], 26, [0]*27)
AA = recDC([1, 5, 21, 10, 25], 63, [0]*64)
end1 = time.time()
print(AA)
print("time is %.4f" % (float(end1)-float(start1)))
print(k)

def dpMakeChange(coinValueList,change,minCoins):
    for cents in range(change+1):
        coinCount = cents
        for j in [c for c in coinValueList if c <= cents]:
            if minCoins[cents-j] + 1 < coinCount:
                coinCount = minCoins[cents-j]+1
        minCoins[cents] = coinCount
    return minCoins[change]

print(dpMakeChange([1,5,10,25],63,[0]*64))


start1 = time.time()
AA = dpMakeChange([1,5,10,25],11,[0]*12)
end1 = time.time()
print(AA)
print("time is %.4f" % (float(end1)-float(start1)))



def dpMakeChange(coinValueList,change,minCoins,coinsUsed):
   for cents in range(change+1):
      coinCount = cents
      newCoin = 1
      for j in [c for c in coinValueList if c <= cents]:
            if minCoins[cents-j] + 1 < coinCount:
               coinCount = minCoins[cents-j]+1
               newCoin = j
      minCoins[cents] = coinCount
      coinsUsed[cents] = newCoin
   return minCoins[change]

def printCoins(coinsUsed,change):
   coin = change
   while coin > 0:
      thisCoin = coinsUsed[coin]
      print(thisCoin)
      coin = coin - thisCoin

def main():
    amnt = 63
    clist = [1,5,10,25]
    # clist = [1,5,10,21,25]
    coinsUsed = [0]*(amnt+1)
    coinCount = [0]*(amnt+1)

    print("Making change for",amnt,"requires")
    print(dpMakeChange(clist,amnt,coinCount,coinsUsed),"coins")
    print("They are:")
    printCoins(coinsUsed,amnt)
    print("The used list is as follows:")
    print(coinsUsed)

main()