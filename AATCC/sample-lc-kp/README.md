# AATCC - 算法分析和复杂性理论 - Analysis of Algorithms and Theory of Computational Complexity

1. Two Sum 兩數之和

5. Longest Palindromic Substring 最长回文子串

7. Reverse Integer 整数反转

13. Roman to Integer 羅馬數字轉整數

15. 3Sum 三数之和

16. 3Sum Closest 整数反转

17. Letter Combinations of a Phone Number 电话号码的字母组合

19. Remove Nth Node From End of List 删除链表的倒数第 N 个结点

20. Valid Parentheses 有效的括号

24. Swap Nodes in Pairs 两两交换链表中的节点

50. Pow(x, n)

53. Maximum Subarray 最长递增子序列

56. Merge Intervals 合并区间

62. Unique Paths 不同路径

63. Unique Paths II 不同路径 II

64. Minimum Path Sum 最小路径和

66. Plus One 加一

69. Sqrt(x) x 的平方根

70. Climbing Stairs 爬楼梯

72. Edit Distance 编辑距离

84. Largest Rectangle in Histogram 柱状图中最大的矩形

85. Maximal Rectangle 最大矩形

100. Same Tree 相同的树

102. Binary Tree Level Order Traversal 二叉树的层序遍历

104. Maximum Depth of Binary Tree 二叉树的最大深度

105. Construct Binary Tree from Preorder and Inorder Traversal 从前序与中序遍历序列构造二叉树

111. Minimum Depth of Binary Tree 二叉树的最小深度

112. Path Sum 路径总和

120. Triangle, 三角形最小路径和

122. Best Time to Buy and Sell Stock II 买卖股票的最佳时机 II

123. Best Time to Buy and Sell Stock III 买卖股票的最佳时机 III

136. Single Number 只出现一次的数字

141. Linked List Cycle 环形链表

144. Binary Tree Preorder Traversal 二叉树的前序遍历

148. Sort List 排序链表

152. Maximum Product Subarray 乘积最大子数组

206. Reverse Linked List 反转链表

208. Implement Trie (Prefix Tree) 实现 Trie (前缀树)

226. Invert Binary Tree 翻转二叉树

232. Implement Queue using Stacks 用栈实现队列

235. Lowest Common Ancestor of a Binary Search Tree 二叉搜索树的最近公共祖先

239. Sliding Window Maximum 滑动窗口最大值

240. Search a 2D Matrix II 搜索二维矩阵 II

242. Valid Anagram 有效的字母异位词

263. Ugly Number 丑数

274. H-Index, H 指数

300. Longest Increasing Subsequence 最长递增子序列

312. Burst Balloons 戳气球

347. Top K Frequent Elements 前 K 个高频元素

374. Guess Number Higher or Lower 二叉树的所有路径

392. Is Subsequence 判断子序列

692. Top K Frequent Words 前 K 个高频单词

720. Longest Word in Dictionary 词典中最长的单词

743. Network Delay Time 网络延迟时间

746. Min Cost Climbing Stairs 爬楼梯的最小损失

787. Cheapest Flights Within K Stops, K 站中转内最便宜的航班

847. Shortest Path Visiting All Nodes 访问所有节点的最短路径

934. Shortest Bridge 最短的桥

997. Find the Town Judge 找到小镇的法官


## LeetCode 1. Two Sum 兩數之和

```
def twoSum(self, nums, target):
    for i in range(len(nums)):
        tmp = nums[i]
        remain = nums[i+1:]
        if target - tmp in remain:
            return[i, remain.index(target - tmp)+ i + 1]

def twoSum(self, nums, target):
    dict = {}
    for i in range(len(nums)):
        if target - nums[i] not in dict:
            dict[nums[i]] = i
        else:
            return [dict[target - nums[i]], i]
```

## LeetCode 5. Longest Palindromic Substring 最长回文子串

```
class Solution:
    def longestPalindrome(self, s: str) -> str:
        if s == s[::-1]:
            return s
        max_len = 1
        ans = s[0]
        for i in range(1, len(s)):
            if i - max_len - 1 >= 0 and s[i - max_len - 1: i + 1] == s[i - max_len - 1: i + 1][::-1]:
                ans = s[i - max_len - 1: i + 1]
                max_len += 2
            if i - max_len >= 0 and s[i - max_len: i + 1] == s[i - max_len: i + 1][::-1]:
                ans = s[i - max_len: i + 1]
                max_len += 1
        return ans
```

## LeetCode 7. Reverse Integer 整数反转

```
class Solution1:
    def reverse(self, x: int) -> int:
        max_32 = 2 ** 31 - 1
        if abs(x) > max_32:
            return 0
        if x < 0:
            rint = -int(str(abs(x))[::-1])
        else:
            rint = int(str(x)[::-1])
        if abs(rint) > max_32:
            return 0
        else:
            return rint 
# 字串
class Solution2:
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        if x==0:
            return 0
        str_x = str(x)
        x = ''
        if str_x[0] == '-':
            x += '-'
        x += str_x[len(str_x)-1::-1].lstrip("0").rstrip("-")
        x = int(x)
        if -2**31<x<2**31-1:
            return x
        return 0
```

## LeetCode 13. Roman to Integer 羅馬數字轉整數

```
class Solution1:
    def romanToInt(self, s):
        rn = {'I':1, 'V':5, 'X':10, 'L':50, 'C':100, 'D':500, 'M':1000}        
        ans=0        
        for i in range(len(s)):            
            if i<len(s)-1 and rn[s[i]]<rn[s[i+1]]:                
                ans -= rn[s[i]]
            else:
                ans += rn[s[i]]
        return ans

class Solution2:
    def romanToInt(self, s: 'str') -> 'int':
        value_roman = {"M":1000, "CM":900, "D":500, "CD": 400,
                       "C":100,"XC":90, "L":50, "XL":40,
                       "X":10, "IX":9, "V":5, "IV":4, "I":1}
        num = 0
        specials_list = ["CM","CD","XC","XL","IX","IV"]
        for i in specials_list:
            if i in s:
                num = num + value_roman[i]
                s=s.replace(i,"")
        for i in s:
            num = num + value_roman[i]
        return(num)
```

## LeetCode 15. 3Sum 三数之和


```
class Solution(object):
    def threeSum(self, nums):
        if len(nums) < 3:
            return[]
        if all (num == 0 for num in nums):
            return [[ 0, 0, 0]]
        found = []
        nums = sorted(nums)
        rightmost = len(nums) - 1
        for index, eachNum in enumerate(nums):
            left = index + 1
            right = rightmost
            while left < right:
                check_sum = (eachNum + nums[left] + nums[right])
                if check_sum == 0:
                    new_found = [eachNum, nums[left], nums[right]]
                    if new_found not in found:
                        found.append(new_found)
                    right -= 1
                elif check_sum < 0:
                    left += 1
                else :
                    right -= 1
        return found

## 複雜度低版本
class Solution2(object):
    def threeSum(self, nums):
        if len(nums) < 3:
            return []
        if all (num == 0 for num in nums):
            return [[0, 0, 0]]
        found = []
        nums = sorted(nums)
        rightmost = len(nums) - 1
        for index, eachNum in enumerate(nums):
            if index > 0 and nums[index] == nums[index - 1]:
                continue
            left = index + 1
            right = rightmost
            while left < right:
                check_sum = (eachNum + nums[left] + nums[right])
                if check_sum == 0:
                    found.append([eachNum, nums[left], nums[right]])
                    left += 1
                    while left < right and nums[left] == nums[left - 1]:
                        left += 1
                elif check_sum < 0:
                    left += 1
                else :
                    right -= 1
        return found
```

## LeetCode 16. 3Sum Closest 整数反转

```
from typing import List
class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        n = len(nums)
        nums.sort()
        re_min = 0 #存储当前最小的差值
        for i in range(n):
            low = i+1
            high = n-1
            while low < high:
                three_sum = nums[i] + nums[low] + nums[high]
                x = target - three_sum #当前三数的差值
                if re_min == 0:
                    re_min = abs(x)
                    sum_min = three_sum #sum_min为当前最接近的和
                if abs(x) < re_min:
                    re_min = abs(x)
                    sum_min = three_sum
                if three_sum == target:
                    return target
                elif three_sum < target:
                    low += 1
                else:
                    high -= 1
        return sum_min
```

## LeetCode 17. Letter Combinations of a Phone Number 电话号码的字母组合


```
class Solution(object):
    def letterCombinations(self, digits):
        """
        动态规划
        """
        if not digits:
            return []
        d = {'2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
             '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'}
        n = len(digits)
        dp = [[] for _ in range(n)]
        dp[0] = [x for x in d[digits[0]]]
        for i in range(1, n):
            dp[i] = [x + y for x in dp[i - 1] for y in d[digits[i]]]
        return dp[-1]

    def letterCombinations2(self, digits):
        """
        使用变量代替上面的列表，降低空间复杂度
        """
        if not digits:
            return []
        d = {'2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
             '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'}
        n = len(digits)
        res = ['']
        for i in range(n):
            res = [x + y for x in res for y in d[digits[i]]]
        return res

    def letterCombinations3(self, digits):
        """
        递归
        """
        d = {'2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
             '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'}
        if not digits:
            return []
        if len(digits) == 1:
            return [x for x in d[digits[0]]]
        return [x + y for x in d[digits[0]] for y in self.letterCombinations3(digits[1:])]

class Solution(object):
    def letterCombinations(self, digits):
        if not digits:
            return []
        d = {'2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl',
             '6': 'mno', '7': 'pqrs', '8': 'tuv', '9': 'wxyz'}
        n = len(digits)
        dp = [[] for _ in range(n)]
        dp[0] = [x for x in d[digits[0]]]
        for i in range(1, n):
            dp[i] = [x + y for x in dp[i - 1] for y in d[digits[i]]]
        return dp[-1]
```

## LeetCode 19. Remove Nth Node From End of List 删除链表的倒数第 N 个结点

```
class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        head_dummy = ListNode()
        head_dummy.next = head

        slow, fast = head_dummy, head_dummy
        while(n!=0): # fast先往前走n步
            fast = fast.next
            n -= 1
        while(fast.next!=None):
            slow = slow.next
            fast = fast.next
        # fast 走到结尾后，slow 的下一个节点为倒数第N个节点
        slow.next = slow.next.next # 删除
        return head_dummy.next

class Solution:
    def removeNthFromEnd(self, head, n):
        fast = slow = head
        for _ in range(n):
            fast = fast.next
        if not fast:
            return head.next
        while fast.next:
            fast = fast.next
            slow = slow.next
        slow.next = slow.next.next
        return head
```

## LeetCode 20. Valid Parentheses 有效的括号

```
class Solution(object):
    def isValid(self, s):
        stack = []
        paren_map = {')': '(', ']':'[', '}':'{'}
        for c in s:
            if c not in paren_map:
                stack.append(c)
            elif not stack or paren_map[c] != stack.pop():
                return False
        return not stack

class Solution:
    def isValid(self, s):
        while '{}' in s or '()' in s or '[]' in s:
            s = s.replace('{}', '')
            s = s.replace('[]', '')
            s = s.replace('()', '')
        return s == ''

# 使用栈
class Solution:
    def isValid(self, s: str) -> bool:
        stack = []        
        for item in s:
            if item == '(':
                stack.append(')')
            elif item == '[':
                stack.append(']')
            elif item == '{':
                stack.append('}')
            elif not stack or stack[-1] != item:
                return False
            else:
                stack.pop()        
        return True if not stack else False

# 使用字典
class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        mapping = {
            '(': ')',
            '[': ']',
            '{': '}'
        }
        for item in s:
            if item in mapping.keys():
                stack.append(mapping[item])
            elif not stack or stack[-1] != item: 
                return False
            else: 
                stack.pop()
        return True if not stack else False
```

## LeetCode 24. Swap Nodes in Pairs 两两交换链表中的节点

两两相邻的元素，翻转链表 `pre->a->b->b.next to pre->b->a->b.next`

```
# Knowledge Point

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

    def __repr__(self):
        if self:
            return "{} -> {}".format(self.val, repr(self.next))

if __name__ == "__main__":
    head = ListNode(1)
    head.next = ListNode(2)
    head.next.next = ListNode(3)
    head.next.next.next = ListNode(4)
    head.next.next.next.next = ListNode(5)
    print(head)
```
```
class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        pre, pre.next = self, head
        while pre.next and pre.next.next:
            a = pre.next
            b = a.next
            pre.next, b.next, a.next = b, a, b.next
            pre = a
        return self.next
```

## LeetCode 50. Pow(x, n)

```
# KP
# LC 50 Pow(x, n)

# 递归
def myPow(x, n):
    if not n:
        return 1
    if n < 0:
        return 1/ myPow(x, -n)
    if n % 2:
        return x * myPow(x, n - 1)
    return myPow(x * x, n / 2)

# 非递归
def myPow2(x, n):
    if n < 0:
        x = 1 / x
        n = -n
    pow = 1
    while n:
        if n & 1:
            pow *= x
        x *= x
        n >>= 1
    return pow
```

## LeetCode 53. Maximum Subarray 最长递增子序列

```
class Solution(object):
    def maxSubArray(self, nums):
        for i in range(1, len(nums)):
            nums[i]= nums[i] + max(nums[i-1], 0)
        return max(nums)
# LC 53  (KP)
class Solution(object):
    def maxSubArray(self, nums):
        maxSeq =[0]*len(nums) 
        maxSeq[0] = nums[0]
        for i in range(1, len(nums)):
            maxSeq[i] = max(maxSeq[i-1]+nums[i], nums[i])
        maximum = max(maxSeq)
        return maximum
# Sample
class Solution(object):
    def maxSubArray(self, nums):
        maximum = min(nums)
        m = 0
        for i in range(len(nums)):
            m = max(m+nums[i], nums[i])
            if m > maximum:
                maximum = m 
        return maximum
# More Sample
class Solution:
    def maxSubArray(self, nums):
        for i in range(1, len(nums)):
            if nums[i - 1] > 0:
                nums[i] += nums[i - 1]
        print(nums)
        return max(nums)
```

## LeetCode 56. Merge Intervals 合并区间

```
from typing import List
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort()   #排序列表，以区间开头升序排列
        ans = [intervals[0]]
        L, R = 1, 0
        while L < len(intervals):
            if ans[R][1] < intervals[L][0]:   #如果区间不重合，直接append
                ans.append(intervals[L])
                L += 1
                R += 1
            else:      #如果区间重合，就合并区间
                ans[R] = [ans[R][0], max(ans[R][1], intervals[L][1])]
                L += 1
        return ans
```

## LeetCode 62. Unique Paths 不同路径

```
# 走方格
# 62
class Solution(object):
    def uniquePaths(self, m, n):
        dp = [[0 for _ in range(n)] for _ in range(m)]
        for index in range(m):
            dp[index][0] = 1
        for index in range(n):
            dp[0][index] = 1
        for index_i in range(1, m): 
            for index_j in range(1, n):
                dp[index_i][index_j] = dp[index_i-1][index_j] + dp[index_i][index_j-1]
        return dp[m-1][n-1]
if __name__ == "__main__":
    print(Solution().uniquePaths(3,2))
    print(Solution().uniquePaths(9,4))
```

## LeetCode 63. Unique Paths II 不同路径 II

```
class Solution(object):
    def uniquePathsWithObstacles(self, obstacleGrid):
        m, n = len(obstacleGrid), len(obstacleGrid[0])
        dp = [[0 for _ in range(n)] for _ in range(m)]

        if obstacleGrid[0][0] == 1 or obstacleGrid[m-1][n-1] == 1: 
            return 0
        dp[0][0] = 1
        for index in range(1, m):
            if obstacleGrid[index][0] == 1:
                dp[index][0] = 0
            else:
                dp[index][0] = dp[index-1][0]
        for index in range(1, n):
            if obstacleGrid[0][index] == 1:
                dp[0][index] = 0
            else:
                dp[0][index] = dp[0][index-1]
        for index_i in range(1, m):
            for index_j in range(1, n):
                if obstacleGrid[index_i][index_j] == 1:
                    dp[index_i][index_j] = 0
                else:
                    dp[index_i][index_j] = dp[index_i-1][index_j] + dp[index_i][index_j-1] 
        return dp[m-1][n-1]
```


## LeetCode 64. Minimum Path Sum 最小路径和

```
class Solution(object):
    def minPathSum(self, grid):
        #此数组用于记忆化搜索
        dp = [[0]*len(grid[0]) for i in range(len(grid))]
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                #在起点的时候
                if (i == 0 and j == 0):
                    dp[i][j] = grid[0][0]
                #在左边缘的时候
                elif (j == 0 and i != 0):
                    dp[i][j] = dp[i - 1][j]  + grid[i][j]
                #在上边缘的时候
                elif (i == 0 and j != 0):
                    dp[i][j] = dp[i][j-1] + grid[i][j]
                # 普遍情况下
                else:
                    dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j]                    
        return dp[len(grid)-1][len(grid[0])-1]
```


## LeetCode 66. Plus One 加一

```
from typing import List
class Solution1:
    def plusOne(self, digits: List[int]) -> List[int]:
        h = ''.join(map(str, digits))
        h = str(int(h) + 1)
        output = []
        for ch in h:
            output.append(int(ch))
        return output

class Solution2:
    def plusOne(self, digits: List[int]) -> List[int]:
        return list(map(int, list(str(int(''.join(map(str, digits))) + 1))))

```

## LeetCode 69. Sqrt(x) x 的平方根

### 二分查找

```
class Solution:
    def mySqrt(self, x):
        """
        :type x: int
        :rtype: int
        """
        if x < 2:
            return x
        left, right = 1, x // 2
        while left <= right:
            mid = left + (right - left) // 2
            if mid > x / mid:
                right = mid - 1
            else:
                left = mid + 1
        return left - 1
```

### 牛頓法

```
def squareroot(input_num):
    root = input_num/2
    for k in range(20):
        root = (1/2)* (root + (input_num/root))
    return root

```

思路總結

1. 二分查找，分成左右區間。

2. 牛頓法


## LeetCode 70. Climbing Stairs 爬楼梯

```
class Solution:
    def climbStairs(self, n):
        prev, current = 0, 1
        for i in range(n):
            prev, current = current, prev + current
        return current
```

思路總結

1. 動態規劃，遞迴公式 f(n-1) + f(n-2)，其結果就是費氏數列。來判斷該值有沒有在字典裡面。相對與第一種課堂範例來的理想。


## LeetCode 72. Edit Distance 编辑距离

```
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        dp = [[0] * (len(word2)+1) for _ in range(len(word1)+1)]
        for i in range(len(word1)+1):
            dp[i][0] = i
        for j in range(len(word2)+1):
            dp[0][j] = j
        for i in range(1, len(word1)+1):
            for j in range(1, len(word2)+1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1]) + 1
        return dp[-1][-1]

class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        n1 = len(word1)
        n2 = len(word2)
        if not n1 or not n2:
            return n1 + n2
        dp=[[0] * (n2+1) for _ in range(n1 +1)]
        for j in range(1, n2 + 1):
            dp[0][j] = dp[0][j-1] + 1
        for i in range(1, n1 + 1):
            dp[i][0] = dp[i-1][0]+ 1
        for i in range(1, n1 + 1):
            for j in range(1, n2 + 1):
                if word1[i-1] == word2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j]= min(dp[i][j-1],dp[i-1][j],dp[i-1][j-1]) + 1
        return dp[-1][-1]

class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        n1 = len(word1)
        n2 = len(word2)
        if not n1 or not n2:
            return n1 + n2
        dp0 = list(range(n2+1))
        dp1=[0]*(n2+1)
        for i in range(n1):
            dp1[0]=i+1
            for j in range(len(word2)):
                if word1[i] == word2[j]:
                    dp1[j+1] = dp0[j]
                else:
                    dp1[j+1] = min(dp0[j+1], dp1[j], dp0[j]) + 1
            dp0 = dp1[:]
        return dp1[-1]
```

## LeetCode 84. Largest Rectangle in Histogram 柱状图中最大的矩形

```
from typing import List
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        stack = [-1]
        heights.append(0)
        n,ans = len(heights),0
        for i in range(n):
            while len(stack) > 1 and heights[stack[-1]] > heights[i]:
                p = stack.pop()
                l,r = stack[-1],i
                ans = max(ans,heights[p] * (r - l - 1))            
            stack.append(i)
        return ans

class Solution(object):
    def largestRectangleArea(self, heights):
        # 定义一个栈 stack
        stack = []
        # 添加两个哨兵
        # 在 heights 的前方和后方设置哨兵节点
        heights.insert(0,0)
        heights.append(0)
        dp=[1]*len(heights)
        stack.append()
        for i in range(1, len(heights)):
            # 当前元素大于栈內最后元素
            if heights [i]>=heights[stack[-1]]:
                stack.append(i)
            # 当前元素小于栈内最后元素,需要把楼内的元素 pop 出来
            else:
                while(heights[stack[-1]]>heights[1]):
                    item=stack.pop()
                    dp[item]=i-stack[-1]-1
                stack.append(i)
        # dp=dp[1:-1]
        for j in range(len(dp)):
            dp[j]=dp[j]*heights[j]
        return max(dp)
```

## LeetCode 85. Maximal Rectangle 最大矩形

```
from typing import List
class Solution:
    def maximalRectangle(self, matrix: List[List[str]]) -> int:
        if not matrix or not matrix[0]:
            return 0
        nums = [int(''.join(row), base=2) for row in matrix] # 先将每一行变成2进制的数字
        ans, N = 0, len(nums)
        for i in range(N):# 遍历每一行，求以这一行为第一行的最大矩形
            j, num = i, nums[i]
            while j < N: # 依次与下面的行进行与运算。
                num = num & nums[j]  # num 中为1的部分，说明上下两行该位置都是1，相当于求矩形的高，高度为j-i+1
                # print('num=',bin(num))
                if not num: # 没有1说明没有涉及第i到第j行的竖直矩形
                    break
                width, curnum = 0, num
                while curnum: 
                    # 将cursum与自己右移一位进行&操作。如果有两个1在一起，那么cursum才为1，相当于求矩形宽度
                    width += 1
                    curnum = curnum & (curnum >> 1)
                    # print('curnum',bin(curnum))
                ans = max(ans, width * (j-i+1))
                # print('i','j','width',i,j,width)
                # print('ans=',ans)
                j += 1
        return ans

class Solution:
    def maximalRectangle(self, matrix) -> int:
        if len(matrix) == 0:
            return 0
        res = 0
        m, n = len(matrix), len(matrix[0])
        heights = [0] * n
        for i in range(m):
            for j in range(n):
                if matrix[i][j] == '0':
                    heights[j] = 0
                else:
                    heights[j] = heights[j] + 1
            res = max(res, self.largestRectangleArea(heights))
        return res

    def largestRectangleArea(self, heights):
        heights.append(0)
        stack = []
        res = 0
        for i in range(len(heights)):
            while stack and heights[i] < heights[stack[-1]]:
                s = stack.pop()
                res = max(res, heights[s] * ((i - stack[-1] - 1) if stack else i))
            stack.append(i)
        return res
```
## LeetCode 100. Same Tree 相同的树

```
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    # def isSameTree(self, p: TreeNode, q: TreeNode) -> bool:
    def isSameTree(self, p, q):
        if not p and not q:
            return True
        elif p is not None and q is not None:
            if p.val == q.val:
                return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
            else:
                return False
        else:
            return False
```

## LeetCode 102. Binary Tree Level Order Traversal 二叉树的层序遍历

```
# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution(object):
    def levelOrder(self, root):
        if not root:
            return []
        queue = [(root, 0)]
        levelMap = {}
        while queue:
            node, level = queue.pop(0)
            if node.left:
                queue.append((node.left, level+1))
            if node.right:
                queue.append((node.right, level+1))
            if level in levelMap:
                levelMap[level].append(node.val)
            else:
                levelMap[level] = [node.val]
        result = []
        for key, value in levelMap.items():
            result.append(value)
        return result
if __name__ == '__main__':
    tree = TreeNode(3)
    tree.left = TreeNode(9)
    tree.right = TreeNode(20)
    tree.right.left = TreeNode(15)
    tree.right.right = TreeNode(7)
    print(Solution().levelOrder(tree))
```

## LeetCode 104. Maximum Depth of Binary Tree 二叉树的最大深度

```
给定二叉树 [3,9,20,null,null,15,7]，

    3
   / \
  9  20
    /  \
   15   7
返回它的最大深度 3 。

```

```
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution:
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if root is None:
            return 0
        else:
            return max(self.maxDepth(root.left), self.maxDepth(root.right)) + 1
if __name__ == '__main__':
    root = TreeNode(3)
    root.left = TreeNode(9)
    root.right = TreeNode(20)
    root.left.left = TreeNode(7)
    root.left.right = TreeNode(15)
    print(Solution().maxDepth(root))
```

## LeetCode 105. Construct Binary Tree from Preorder and Inorder Traversal 从前序与中序遍历序列构造二叉树

### LC 105 說明

```

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution(object):
    def buildTree(self, preorder, inorder):
        # ... Code ...
        return root
preorder = [3,9,20,15,7]
inorder = [9,3,15,20,7]
root = Solution().buildTree(preorder, inorder)

```

前序遍历：遍历顺序为 父节点 -> 左子节点 -> 右子节点

中序遍历：遍历顺序为 左子节点 -> 父节点 -> 右子节点

前序遍历的第一个元素为根节点，而在中序遍历中，该根节点所在位置的左侧为左子树，右侧为右子树。

```
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        dic = {}
        for idx,i in enumerate(inorder):
            dic[i] = idx

        def dfs(pre_left,ino_l,ino_r):
            if pre_left>=len(preorder) or ino_l> ino_r or ino_r>=len(inorder) or ino_l<0 or ino_r<0: return None
            node = TreeNode(preorder[pre_left])
            mid = dic[preorder[pre_left]]
           # print(mid)
            pre_left = pre_left 
            node.left = dfs(pre_left + 1,ino_l,mid-1)
            node.right = dfs(pre_left + (mid - ino_l+1), mid+1, ino_r )
            return node
        return dfs(0,0,len(inorder)-1)

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution(object):
    def buildTree(self, preorder, inorder):
        if len(inorder) == 0:
            return None
        # 前序遍历第一个值为根节点
        root = TreeNode(preorder[0])
        # 因为没有重复元素，所以可以直接根据值来查找根节点在中序遍历中的位置
        mid = inorder.index(preorder[0])
        # 构建左子树
        root.left = self.buildTree(preorder[1:mid + 1], inorder[:mid])
        # 构建右子树
        root.right = self.buildTree(preorder[mid + 1:], inorder[mid + 1:])
        return root
preorder = [3,9,20,15,7]
inorder = [9,3,15,20,7]
root = Solution().buildTree(preorder, inorder)
```

## LeetCode 111. Minimum Depth of Binary Tree 二叉树的最小深度

```
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution:
    def minDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if root is None:
            return 0
        if root.left and root.right:
            return min(self.minDepth(root.left), self.minDepth(root.right)) + 1
        else:
            return max(self.minDepth(root.left), self.minDepth(root.right)) + 1
if __name__ == '__main__':
    root = TreeNode(3)
    root.left = TreeNode(9)
    root.right = TreeNode(20)
    root.right.left = TreeNode(7)
    root.right.right = TreeNode(15)
    print(Solution().minDepth(root))
```

## LeetCode 120. Triangle, 三角形最小路径和

```
from typing import List
class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        depth = len(triangle)
        for i in range(-2, -depth-1, -1):
            for j in range(depth + 1 + i):
                triangle[i][j] += min(triangle[i+1][j], triangle[i+1][j+1])
        return triangle[0][0]
```

## LeetCode 112. Path Sum 路径总和

```
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def hasPathSum(self, root: TreeNode, targetSum: int) -> bool:
        if not root: return False
        res = []
        def pathval(root, path):
            path = path.copy()
            if not (root.left or root.right): 
                res.append(path)
                return
            if root.left:
                path.append(root.left.val)
                pathval(root.left, path)
                path.pop()
            if root.right:
                path.append(root.right.val)
                pathval(root.right, path)
                path.pop()
        pathval(root, [root.val])
        for path in res: 
            if sum(path) == targetSum: return True
        return False
```

## LeetCode 122. Best Time to Buy and Sell Stock II 买卖股票的最佳时机 II

```
# 122(KP) 
class Solution:
    def maxProfit (self, prices):
        if len(prices) <= 1:
            return 0
        total = 0
        for i in range(1, len(prices)):
            if prices[i] > prices[i-1]:
                total += prices[i] - prices[i-1]
        return total
if __name__ == '__main__':
    # prices = [ 6, 1, 3, 2, 4, 7]
    prices = [7, 1, 5, 3, 6,4]
    # prices = [1, 2, 3, 4, 5]
    print(Solution().maxProfit(prices))
```

## LeetCode 123. Best Time to Buy and Sell Stock III 买卖股票的最佳时机 III

```
class Solution:
    def maxProfit(self, prices):
        if prices==[]:
            return 0
        length=len(prices)
        #结束时的最高利润=[天数][是否持有股票][卖出次数] 
        dp=[ [[0,0,0],[0,0,0] ] for i in range(0,length) ]
        #第一天休息
        dp[0][0][0]=0 
        #第一天买入
        dp[0][1][0]=-prices[0]
        # 第一天不可能已经有卖出
        dp[0][0][1] = float('-inf') 
        dp[0][0][2] = float('-inf')
        #第一天不可能已经卖出
        dp[0][1][1]=float('-inf')
        dp[0][1][2]=float('-inf')
        for i in range(1,length):
            #未持股，未卖出过，说明从未进行过买卖
            dp[i][0][0]=0 
            #未持股，卖出过1次，可能是今天卖的，可能是之前卖的
            dp[i][0][1]=max(dp[i-1][1][0]+prices[i],dp[i-1][0][1]) 
            #未持股，卖出过2次，可能是今天卖的，可能是之前卖的
            dp[i][0][2]=max(dp[i-1][1][1]+prices[i],dp[i-1][0][2]) 
            #持股，未卖出过，可能是今天买的，可能是之前买的
            dp[i][1][0]=max(dp[i-1][0][0]-prices[i],dp[i-1][1][0]) 
            #持股，卖出过1次，可能是今天买的，可能是之前买的
            dp[i][1][1]=max(dp[i-1][0][1]-prices[i],dp[i-1][1][1]) 
            #持股，卖出过2次，不可能
            dp[i][1][2]=float('-inf')
        return max(dp[length-1][0][1],dp[length-1][0][2],0)
if __name__ == "__main__":
    list = [3,1,5,2,1,3,1,9] 
    print(Solution().maxProfit(list))
```

## LeetCode 136. Single Number 只出现一次的数字

```
class Solution:
    def singleNumber(self, nums):
        a = 0
        for num in nums:
            a = a ^ num
        return a

from typing import List
class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        return functools.reduce(int.__xor__,nums)
```

## LeetCode 141. Linked List Cycle 环形链表

```
# Linked List Cycle
class Solution(object):
    def hasCycle(self, head):
        fast, slow = head, head
        while fast and fast.next:
            fast, slow = fast.next.next, slow.next
            if fast == slow:
                return True
        return False
```

## LeetCode 144. Binary Tree Preorder Traversal 二叉树的前序遍历

```
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
# Python 迭代
class Solution(object):
    def preorderTraversal(self, root):
        if not root:
            return []
        
        stack = [root]
        res = []
        while stack:
            cur = stack.pop()
            res.append(cur.val)            
            if cur.right:
                stack.append(cur.right)
            if cur.left:
                stack.append(cur.left)
        return res

# Python 递归
class Solution(object):
    def preorderTraversal(self, root):
        if not root:
            return []
        return [root.val] + self.preorderTraversal(root.left) + self.preorderTraversal(root.right)

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution(object):
    def preorderTraversal(self, root):
        ret = []
        stack = [root]
        while stack:
            node = stack.pop()
            if node:
                ret.append(node.val)
                stack.append(node.right)
                stack.append(node.left)
        return ret

root = TreeNode(5)
root.left = TreeNode(4)
root.right = TreeNode(8)
root.right.left = TreeNode(13)
root.right.right = TreeNode(4)
root.right.right.right = TreeNode(1)
root.left.left = TreeNode(11)
root.left.left.left = TreeNode(7)
root.left.left.right = TreeNode(2)
print(Solution().preorderTraversal(root))
```

## LeetCode 148. Sort List 排序链表

```
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def sortList(self, head: ListNode) -> ListNode:
        h_head = ListNode(-1, head)
        mem = []
        while(head is not None):
            next_h = head.next
            head.next = None
            mem.append(head)
            head = next_h
        mem = sorted(mem, key=lambda x: x.val)
        n = len(mem)
        if n == 0:
            return None
        h_head.next = mem[0]
        for i in range(n-1):
            mem[i].next = mem[i+1]     
        return h_head.next
```

## LeetCode 152. Maximum Product Subarray 乘积最大子数组

```
class Solution:
    def maxProduct(self, A):
        B = A[::-1]
        for i in range(1, len(A)):
            A[i] *= A[i - 1] or 1
            B[i] *= B[i - 1] or 1
        return max(max(A),max(B)) 
```

## LeetCode 206. Reverse Linked List 反转链表

### 解题思路

两种思路 1.后挂 2.交换

```
# Knowledge Point

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

    def __repr__(self):
        if self:
            return "{} -> {}".format(self.val, repr(self.next))

    if __name__ == "__main__":
        head = ListNode(1)
        head.next = ListNode(2)
        head.next.next = ListNode(3)
        head.next.next.next = ListNode(4)
        head.next.next.next.next = ListNode(5)
        print(head)
```

```
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

    def __repr__(self):
        if self:
            return "{} -> {}".format(self.val, repr(self.next))

class Solution:
    def reverseList(self, head):
        dummy = ListNode(float("-inf"))
        while head:
            dummy.next, head.next, head = head, dummy.next, head.next
        return dummy.next

# 交換法

class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = None
        while head:
            next = head.next
            head.next = prev
            prev = head
            head = next
        return prev

# 遞迴法
class Solution:
    def reverseList(self, head: Optional[ListNode], prev=None) -> Optional[ListNode]:
        if not head: return prev
        next = head.next # 先把下一個記起來
        head.next = prev # 將自己反過來指向前一個
        return self.reverseList(next, head)
```

## LeetCode 208. Implement Trie (Prefix Tree) 实现 Trie (前缀树)

```
class TreeNode(object):
    def __init__(self):
        self.word = False
        self.children = {}
class Trie(object):
    def __init__(self):
        self.root = TreeNode()
    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TreeNode()
            node = node.children[char]
        node.word = True
    def search(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.word
    def startsWith(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True
```

## LeetCode 226. Invert Binary Tree 翻转二叉树

```
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def invertTree(self, root: TreeNode) -> TreeNode:
        self.spyxfamily(root)
        return root
    def spyxfamily(self,root):
        if root is  None:
            return
        root.left,root.right=root.right,root.left
        self.spyxfamily(root.left)
        self.spyxfamily(root.right)

# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = No
class Solution(object):
    def invertTree(self, root):
        if root:
            root.left, root.right = self.invertTree(root.right), self.invertTree(root.left)
            return root
```

## LeetCode 232. Implement Queue using Stacks 用栈实现队列

### 解题思路

- 用栈实现一个队列的基本操作：push(x)、pop()、peek()、empty()。

```
class MyQueue:
    def __init__(self):
        """
        in主要负责push，out主要负责pop
        """
        self.stack_in = []
        self.stack_out = []
    def push(self, x: int) -> None:
        """
        有新元素进来，就往in里面push
        """
        self.stack_in.append(x)
    def pop(self) -> int:
        """
        Removes the element from in front of queue and returns that element.
        """
        if self.empty():
            return None
        
        if self.stack_out:
            return self.stack_out.pop()
        else:
            for i in range(len(self.stack_in)):
                self.stack_out.append(self.stack_in.pop())
            return self.stack_out.pop()
    def peek(self) -> int:
        """
        Get the front element.
        """
        ans = self.pop()
        self.stack_out.append(ans)
        return ans
    def empty(self) -> bool:
        """
        只要in或者out有元素，说明队列不为空
        """
        return not (self.stack_in or self.stack_out)

class MyQueue:
    def __init__(self):
        self.A, self.B =[], []
    def push (self, x):
        self.A.append(x)

    def pop(self):
        self.peek()
        return self.B.pop()

    def peek(self):
        if not self.B:
            while self.A:
                self.B.append(self.A.pop())
        return self.B[-1]

    def empty(self):
        return not self.A and not self.B
```

## LeetCode 235. Lowest Common Ancestor of a Binary Search Tree 二叉搜索树的最近公共祖先

```
class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        if p.val<root.val and q.val<root.val:
            return self.lowestCommonAncestor(root.left,p,q)
        if p.val>root.val and q.val>root.val:
            return self.lowestCommonAncestor(root.right,p,q)
```

## LeetCode 239. Sliding Window Maximum 滑动窗口最大值

```
class Solution(object):
    def maxSlidingWindow(self, nums, k):
        win, ret = [], []
        for i, v in enumerate(nums):
            if i >= k and win[0] <= i - k: win.pop(0)
            while win and nums[win[-1]] <= v: win.pop()
            win.append(i)
            if i >= k - 1: ret.append(nums[win[0]])
        return ret
# 思路：维护窗口，向右移动时左侧超出窗口的值弹出，
# 因为需要的是窗口内的最大值，
# 所以只要保证窗口内的值是递减的即可，小于新加入的值全部弹出。
# 最左端即为窗口最大值 python解法：
x = [1,3,-1,-3,5,3,6,7]
kn = 3
ob = Solution()
print(ob.maxSlidingWindow(x, kn))
```

## LeetCode 240. Search a 2D Matrix II 搜索二维矩阵 II

```
class Solution:
    def searchMatrix(self, matrix, target):
        m = len(matrix)
        if m == 0:
            return False
        n = len(matrix[0])
        if n == 0:
            return False

        i = m - 1
        j = 0
        while i >= 0 and j < n:
            if matrix[i][j] == target:
                return True
            elif matrix[i][j] < target:
                j = j + 1
            else:
                i = i - 1
        return False
```

## LeetCode 242. Valid Anagram 有效的字母异位词

```
# defaultdict 解
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        from collections import defaultdict
        s_dict = defaultdict(int)
        t_dict = defaultdict(int)
        for x in s:
            s_dict[x] += 1
        for x in t:
            t_dict[x] += 1
        return s_dict == t_dict

# dic 解
class Solutiont:
    def isAnagram(self, s, t):
        dic1, dic2 = {}, {}
        for item in s:
            dic1[item] = dic1.get(item, 0) + 1
        for item in t:
            dic2[item] = dic2.get(item, 0) + 1
        return dic1 == dic2

# ASCII 解
class Solutiont:
    def isAnagram(self, s, t):
        dic1, dic2 = [0]*26, [0]*26
        for item in s:
            dic1[ord(item) - ord('a')] += 1
        for item in t:
            dic2[ord(item) - ord('a')] += 1
        return dic1 == dic2

class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        return sorted(s) == sorted(t)

```

## LeetCode 263. Ugly Number 丑数

```
class Solution:
    def isUgly(self, num):
        if num == 0:
            return False
        for i in [2,3,5]:
            while num % i == 0:
                num /= i
        return num == 1
```

## LeetCode 274. H-Index, H 指数

```
class Solution(object):
    def hIndex(self, citations):
        index = 0
        citations.sort(reverse=True)
        for i in citations:
            if i > index:
                index +=1
        return index
```

## LeetCode 300. Longest Increasing Subsequence 最长递增子序列

```
class Solution(object):
    def lengthOfLIS(self, nums):
        if not nums:
            return 0
        N = len(nums)
        dp = [1 for _ in range(N)]
        ans = 1
        for i in range(1, N):
            temp = []
            temp.append(1)
            for j in range(i):
                if nums[i] > nums[j]:
                    temp.append(dp[j] + 1)
            dp[i] = max(temp)
            ans = max(ans, dp[i]) 
        return ans

# Sample 简洁版
class Solution(object):
    def lengthOfLIS(self, nums):
        if not nums:
            return 0
        N = len(nums)
        dp = [1 for _ in range(N)]
        ans = 1
        for i in range(1, N):
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
            ans = max(ans, dp[i])
        return ans
```

## LeetCode 312. Burst Balloons 戳气球

```
from typing import List
def maxCoins(self, nums: List[int]) -> int:
        if len(nums) == 0:
            return 0
        if len(nums) < 2:
            return nums[0]
        nums = [1] + nums + [1]
        dp = [[0] * len(nums) for _ in range(len(nums))]
        for i in range(len(nums) - 1, -1, -1):
            for j in range(i + 2, len(nums)):
                for k in range(i + 1, j):
                    dp[i][j] = max(dp[i][j], dp[i][k] + dp[k][j] + nums[i] * nums[k] * nums[j])
        return dp[0][-1]

class Solution:
    def maxCoins(self, nums):
        if not nums: return 0
        if len(nums) == 1: return nums[0]
        nums = [1] + nums + [1]
        dp = [[0] * len(nums) for _ in range(len(nums))]
        for i in range(len(nums)-1, -1, -1): 
            for j in range(i+2, len(nums)):
                for k in range(i+1, j):
                    dp[i][j] = max(dp[i][j], dp[i][k]+dp[k][j]+nums[i]*nums[k]*nums[j])
        return dp[0][-1]

class Solution:
    def maxCoins(self, nums): 
        n = len(nums) 
        nums.insert(0, 1)
        nums.append(1)
        c = [[0] * (n + 2) for _ in range(n + 2)]
        for len_ in range(1, n + 1):
            for left in range(1, n - len_ + 2):
                right = left + len_ - 1
                for k in range(left, right + 1):
                    c[left][right] = max(c[left][right], c[left][k - 1] + nums[left - 1] * nums[k] * nums[right + 1] + c[k + 1][right])
        return c[1][n]
```
## LeetCode 347. Top K Frequent Elements 前 K 个高频元素

```
from typing import List 
#时间复杂度：O(nlogk)
#空间复杂度：O(n)
import heapq
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        #要统计元素出现频率
        map_ = {} #nums[i]:对应出现的次数
        for i in range(len(nums)):
            map_[nums[i]] = map_.get(nums[i], 0) + 1
        
        #对频率排序
        #定义一个小顶堆，大小为 k
        pri_que = [] #小顶堆
        
        #用固定大小为 k 的小顶堆，扫面所有频率的数值
        for key, freq in map_.items():
            heapq.heappush(pri_que, (freq, key))
            if len(pri_que) > k: #如果堆的大小大于了K，则队列弹出，保证堆的大小一直为k
                heapq.heappop(pri_que)
        
        #找出前 K 个高频元素，因为小顶堆先弹出的是最小的，所以倒叙来输出到数组
        result = [0] * k
        for i in range(k-1, -1, -1):
            result[i] = heapq.heappop(pri_que)[1]
        return result
```

## LeetCode 374. Guess Number Higher or Lower 二叉树的所有路径

```
class Solution:
    def guessNumber(self, n: int) -> int:
        left ,right = 1,n
        while left <= right:
            mid = (left + right) // 2
            if guess(mid) == 1:
                left = mid + 1
            elif guess(mid) == -1:
                right = mid - 1
            else :
                return mid
```

## LeetCode 392. Is Subsequence 判断子序列

```
class Solution:
    def isSubsequence(self, s, t):
        if not s:
            return True
        i, l_s = 0, len(s)
        for v in t:
            if s[i] == v:
                i += 1
            if i == l_s:
                return True
        return False
```

## LeetCode 692. Top K Frequent Words 前 K 个高频单词

```
from typing import List
class Solution:
    def topKFrequent(self, words: List[str], k: int) -> List[str]:
        q = []
        dic = collections.defaultdict(int)
        for word in words:
            dic[word] += 1
        
        for key, val in dic.items():
            heapq.heappush(q, (-val, key))
        
        return [heapq.heappop(q)[1] for i in range(k)]
```

## LeetCode 720. Longest Word in Dictionary 词典中最长的单词

```
class Solution(object):
    def longestWord(self, words):
        valid = set([""])
        for word in sorted(words, key=len):
            if word[:-1] in valid:
                valid.add(word)
        return max(sorted(valid), key=len)
```

## LeetCode 743. Network Delay Time 网络延迟时间

```
from typing import List
class Solution:
    # Bellman-Ford 算法
    def networkDelayTime(self, times: List[List[int]], n: int, k: int) -> int:
        dis={node:float('inf') for node in range(1,n+1)}
        dis[k]=0
        for _ in range(n-1):
            for u,v,w in times:
                dis[v]=min(dis[v],dis[u]+w)
        res=max(dis.values())
        return res if res != float('inf') else -1   
```

## LeetCode 746. Min Cost Climbing Stairs 爬楼梯的最小损失

```
class Solution:
    def minCostClimbingStairs(self, cost):
        cost.append(0)
        for i in range(2, len(cost)):
            cost[i] += min(cost[i - 1], cost[i - 2])
        return cost[-1]
```

## LeetCode 787. Cheapest Flights Within K Stops, K 站中转内最便宜的航班

```
class Solution:
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, K: int) -> int:
        if src == dst: return 0
        graph = collections.defaultdict(dict)
        for start,end,cost in flights:
            graph[start][end] = cost
        queue = [(0,0,src)]
        while queue:
            cost, k, end = heapq.heappop(queue)
            if k > K+1 : continue
            if end == dst: return cost
            for key, val in graph[end].items():
                heapq.heappush(queue,(cost+val,k+1,key))
        return -1
```

## LeetCode 847. Shortest Path Visiting All Nodes 访问所有节点的最短路径

```
class Solution:
    def shortestPathLength(self, graph: List[List[int]]) -> int:
        q = collections.deque([])
        visited = set()
        n = len(graph)
        for i in range(n):
            q.append((i, 1 << i))
            visited.add((i, 1 << i))
        dis = 0
        while q:
            dis += 1
            for _ in range(len(q)):
                cur, cur_state = q.popleft()
                for nxt in graph[cur]:
                    nxt_state = cur_state | (1 << nxt)
                    if nxt_state == (1 << n) - 1: return dis
                    if (nxt, nxt_state) not in visited:
                        q.append((nxt, nxt_state))
                        visited.add((nxt, nxt_state))
        return 0
```

## LeetCode 934. Shortest Bridge 最短的桥

```
from collections import deque
from typing import List
class Solution:
    def shortestBridge(self, grid: List[List[int]]) -> int:
        def dfs(grid, x, y):
            grid[x][y] = 0
            seen.append([x, y])
            seen_set.add(f'{x}#{y}')
            axis = [[x - 1, y], [x + 1, y], [x, y -1], [x, y + 1]]
            for x, y in axis:
                if 0 <= x < m and 0 <= y < n and grid[x][y] == 1:
                    dfs(grid, x, y)
        
        def bfs(grid, seen):
            seen = deque(seen)
            seen_other_flag = False
            level = 0
            while seen:
                for _ in range(len(seen)):
                    x, y = seen.popleft()
                    axis = [[x - 1, y], [x + 1, y], [x, y -1], [x, y + 1]]
                    for x, y in axis:
                        index = f'{x}#{y}'
                        if 0 <= x < m and 0 <= y < n and index not in seen_set:
                            if grid[x][y] == 0:
                                seen.append([x, y])
                                seen_set.add(f'{x}#{y}')
                            else:
                                return level
                level += 1
            return level
        seen = []
        seen_set = set()
        m = len(grid)
        n = len(grid[0])
        search_flag = 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1 and not search_flag:
                    dfs(grid, i, j)
                    search_flag = 1
        level = bfs(grid, seen)
        return level
```

## LeetCode 997. Find the Town Judge 找到小镇的法官

```
class Solution(object):
    def findJudge(self, N, trust):
        if not trust:
            return 1
        mapping = {}
        unique = set()
        for truste_list in trust:
            unique.add(truste_list[0])
            if truste_list[1] in mapping:
                mapping[truste_list[1]] += 1
            else:
                mapping[truste_list[1]] = 1

        unique_set = len(unique)
        for key, value in mapping.items():
            if (value == N-1) & (unique_set == N-1):
                return key
        return -1

    def findJudge2(self, N, trust):
        from collections import Counter
        people = set([x[0] for x in trust])
        if not len(people):
            if N == 1:
                return 1
            else:
                return -1

        if len(people) == N - 1:
            trustee = Counter([x[1] for x in trust])
            for t in trustee.keys():
                if trustee[t] == N - 1:
                    return t
            return -1
        return -1
```
