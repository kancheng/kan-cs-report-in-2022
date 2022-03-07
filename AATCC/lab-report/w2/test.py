import time
# 計算通過 EX 1 的效率
start = time.process_time()
class Solution(object):
   def twoSum(self, nums, target):
      for i in range(len(nums)):
         tmp = nums[i]
         remain = nums[i+1:]
         if target - tmp in remain:
                return[i, remain.index(target - tmp)+ i + 1]
input_list = [ 2, 7, 11, 15]
target = 9
ob1 = Solution()
print(ob1.twoSum(input_list, target))
end = time.process_time()
print("Process Time: time of EX 1 is %.5f" % float(end-start))

start = time.perf_counter()
class Solution(object):
   def twoSum(self, nums, target):
      for i in range(len(nums)):
         tmp = nums[i]
         remain = nums[i+1:]
         if target - tmp in remain:
                return[i, remain.index(target - tmp)+ i + 1]
input_list = [ 2, 7, 11, 15]
target = 9
ob1 = Solution()
print(ob1.twoSum(input_list, target))
end = time.perf_counter()
print("Perf Counter: time of EX 1 is %.5f" % float(end-start))

# 計算通過 EX 2 的效率
start = time.process_time()
class Solution(object):
    def twoSum(self, nums, target):
        dict = {}
        for i in range(len(nums)):
            if target - nums[i] not in dict:
                dict[nums[i]] = i
            else:
                return [dict[target - nums[i]], i]
input_list = [ 2, 7, 11, 15]
target = 9
ob1 = Solution()
print(ob1.twoSum(input_list, target))
end = time.process_time()
print("Process Time: time of EX 2 is %.5f" % float(end-start))

start = time.perf_counter()
class Solution(object):
    def twoSum(self, nums, target):
        dict = {}
        for i in range(len(nums)):
            if target - nums[i] not in dict:
                dict[nums[i]] = i
            else:
                return [dict[target - nums[i]], i]
input_list = [ 2, 7, 11, 15]
target = 9
ob1 = Solution()
print(ob1.twoSum(input_list, target))
end = time.perf_counter()
print("Perf Counter: time of EX 2 is %.5f" % float(end-start))