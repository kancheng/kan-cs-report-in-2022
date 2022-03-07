import time
from typing import List
# 計算通過 EX 1 的效率
start = time.process_time()
class Solution1:
    def plusOne(self, digits: List[int]) -> List[int]:
        h = ''.join(map(str, digits))
        h = str(int(h) + 1)
        output = []
        for ch in h:
            output.append(int(ch))
        return output
x = [1,2,3]
ob1 = Solution1()
print(ob1.plusOne([1,2,3]))
end = time.process_time()
print("Process Time: time of EX 1 is %.5f" % float(end-start))

start = time.perf_counter()
class Solution1:
    def plusOne(self, digits: List[int]) -> List[int]:
        h = ''.join(map(str, digits))
        h = str(int(h) + 1)
        output = []
        for ch in h:
            output.append(int(ch))
        return output
x = [1,2,3]
ob1 = Solution1()
print(ob1.plusOne([1,2,3]))
end = time.perf_counter()
print("Perf Counter: time of EX 1 is %.5f" % float(end-start))

# 計算通過 EX 2 的效率
start = time.process_time()
class Solution2:
    def plusOne(self, digits: List[int]) -> List[int]:
        return list(map(int, list(str(int(''.join(map(str, digits))) + 1))))
x = [1,2,3]
ob2 = Solution2()
print(ob2.plusOne([1,2,3]))
end = time.process_time()
print("Process Time: time of EX 2 is %.5f" % float(end-start))

start = time.perf_counter()
class Solution2:
    def plusOne(self, digits: List[int]) -> List[int]:
        return list(map(int, list(str(int(''.join(map(str, digits))) + 1))))
x = [1,2,3]
ob2 = Solution2()
print(ob2.plusOne([1,2,3]))
end = time.perf_counter()
print("Perf Counter: time of EX 2 is %.5f" % float(end-start))