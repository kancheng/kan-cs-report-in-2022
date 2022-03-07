# AATCC - 算法分析和复杂性理论 - Analysis of Algorithms and Theory of Computational Complexity

> 2101212850 干皓丞

PKU 2022 個人實驗報告作業


## Details

針對算法分析和复杂性理论所做的針對性考試筆記。

## Log

1. W1 : 

- LeetCode : `1. Two Sum 兩數之和` , `69. Sqrt(x) x 的平方根`, `70. Climbing Stairs 爬楼梯`.

- Knowledge Point : LeetCode 1, 69, 70.

2. W2 : 

- LeetCode : `7. Reverse Integer 整数反转`, `13. Roman to Integer 羅馬數字轉整數`, `66. Plus One 加一`.

- Knowledge Point : `平方根函數`, `＊牛頓法`,  - `乱序字符串检查 - 逐字检查`, `乱序字符串检查 - 排序与比较`, `＊乱序字符串检查 - 计数与比较(ASCII)`, `＊栈 - 十进制转二进制`, `栈 - 2 ~ 16 进制轉換`

3. W3 :



## LeetCode 1. Two Sum 兩數之和

Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

You can return the answer in any order.

给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出 和为目标值 target  的那 两个 整数，并返回它们的数组下标。

你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。

你可以按任意顺序返回答案。

```
class Solution(object):
   def twoSum(self, nums, target):
      """
      :type nums: List[int]
      :type target: int
      :rtype: List[int]
      """
      for i in range(len(nums)):
         tmp = nums[i]
         remain = nums[i+1:]
         if target - tmp in remain:
                return[i, remain.index(target - tmp)+ i + 1]
input_list = [ 2, 7, 11, 15]
target = 9
ob1 = Solution()
print(ob1.twoSum(input_list, target))
```

```
class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
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
```

思路總結

1. 用 For 將每個元素讀過一遍，然後將其逐一取出來一個個判斷，若目標為 9，找到元素 2 ，就會找 7，若找到元素 7 ，就會找 2。效率上沒有很理想。

2. 運用 Python 的字典可以直接去找。用 For 去找，剩下用 IF 來判斷該值有沒有在字典裡面。相對與第一種課堂範例來的理想。


## LeetCode 7. Reverse Integer 整数反转

Given a signed 32-bit integer x, return x with its digits reversed. If reversing x causes the value to go outside the signed 32-bit integer range $[-2^{31}, 2^{31} - 1]$, then return 0.

Assume the environment does not allow you to store 64-bit integers (signed or unsigned).


給你一個 32 位的有符號整數 x ，返回將 x 中的數字部分反轉後的結果。

如果反轉後整數超過 32 位的有符號整數的範圍 $[−2^{31}, 2^{31} − 1]$ ，就返回 0。

假設環境不允許存儲 64 位整數（有符號或無符號）。

給定一個整數數組 nums 和一個整數目標值 target，請你在該數組中找出 和為目標值 target  的那 兩個 整數，並返回它們的數組下標。

你可以假設每種輸入只會對應一個答案。但是，數組中同一個元素在答案裡不能重複出現。

你可以按任意順序返回答案。

Example 1:

```
Input: x = 123
Output: 321
```

Example 2:

```
Input: x = -123
Output: -321
```

Example 3:

```
Input: x = 120
Output: 21
```

思路總結

1. 要求反轉 10 進位。

2. 其反轉的數字必須在 $[−2^31, 2^31 − 1]$ 範圍內，若超過範圍則必須輸出 0 

```
# 絕對值
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
x1 = -123
x2 = 123
x3 = 120
ob1 = Solution1()
print(ob1.reverse(x1))
print(ob1.reverse(x2))
print(ob1.reverse(x3))

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
x1 = -123
x2 = 123
x3 = 120
ob2 = Solution2()
print(ob2.reverse(x1))
print(ob2.reverse(x2))
print(ob2.reverse(x3))
```

## LeetCode 13. Roman to Integer 羅馬數字轉整數


Roman numerals are represented by seven different symbols: I, V, X, L, C, D and M.

```
Symbol       Value
I             1
V             5
X             10
L             50
C             100
D             500
M             1000
```

For example, 2 is written as II in Roman numeral, just two one's added together. 

12 is written as XII, which is simply X + II. 

The number 27 is written as XXVII, which is XX + V + II.

Roman numerals are usually written largest to smallest from left to right. 

However, the numeral for four is not IIII. 

Instead, the number four is written as IV. 

Because the one is before the five we subtract it making four. 

The same principle applies to the number nine, which is written as IX. There are six instances where subtraction is used:

I can be placed before V (5) and X (10) to make 4 and 9.
X can be placed before L (50) and C (100) to make 40 and 90.
C can be placed before D (500) and M (1000) to make 400 and 900.
Given a roman numeral, convert it to an integer.

羅馬數字包含以下七種字符: I， V， X， L，C，D 和 M。

例如， 羅馬數字 2 寫做 II ，即為兩個並列的 1 。 12 寫做 XII ，即為 X + II 。 27 寫做 XXVII, 即為 XX + V + II 。

通常情況下，羅馬數字中小的數字在大的數字的右邊。但也存在特例，例如 4 不寫做 IIII，而是 IV。數字 1 在數字 5 的左邊，所表示的數等於大數 5 減小數 1 得到的數值 4 。同樣地，數字 9 表示為 IX。這個特殊的規則只適用於以下六種情況：

I 可以放在 V (5) 和 X (10) 的左邊，來表示 4 和 9。

X 可以放在 L (50) 和 C (100) 的左邊，來表示 40 和 90。

C 可以放在 D (500) 和 M (1000) 的左邊，來表示 400 和 900。

給定一個羅馬數字，將其轉換成整數。


Example 1:

```
Input: s = "III"
Output: 3
Explanation: III = 3.
```

Example 2:

```
Input: "IV"
Output: 4
```

Example 3:

```
Input: "IX"
Output: 9
```

Example 4:

```
Input: "LVIII"
Output: 58
Explanation: L = 50, V= 5, III = 3.
```

Example 5:

```
Input: "MCMXCIV"
Output: 1994
Explanation: M = 1000, CM = 900, XC = 90 and IV = 4.
```

Constraints:

- 1 <= s.length <= 15

- s contains only the characters ('I', 'V', 'X', 'L', 'C', 'D', 'M').

s 僅含字符 ('I', 'V', 'X', 'L', 'C', 'D', 'M')

- It is guaranteed that s is a valid roman numeral in the range $[1, 3999]$.

題目數據保證 s 是一個有效的羅馬數字，且表示整數在範圍 $[1, 3999]$ 內

題目所給測試用例皆符合羅馬數字書寫規則，不會出現跨位等情況。 IL 和 IM 這樣的例子並不符合題目要求，49 應該寫作 XLIX，999 應該寫作 CMXCIX 。關於羅馬數字的詳盡書寫規則，可以參考 羅馬數字 - Mathematics 。

https://b2b.partcommunity.com/community/knowledge/en/detail/10753/Roman\%20numerals

思路總結

1. 給一個羅馬數字，將其轉換成整數。其輸入確保在 1 ~ 3999 的範圍內。

2. 按要求的羅馬數字計算出對應的 10 進位阿拉伯數字即可。

3. 首先建立一個 HashMap 或者是 Python 的字典來抓相對應的符號與值，而後對字串從左至右來處理。若當前字符代表的值不小於右邊，就加上該值；否則就减去該值。類推到最左邊的數。

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
x = "III"
ob1 = Solution1()
print(ob1.romanToInt(x))
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
x = "III"
ob2 = Solution2()
print(ob2.romanToInt(x))
```


## LeetCode 69. Sqrt(x) x 的平方根

Given a non-negative integer x, compute and return the square root of x.

Since the return type is an integer, the decimal digits are truncated, and only the integer part of the result is returned.

Note: You are not allowed to use any built-in exponent function or operator, such as pow(x, 0.5) or x ** 0.5.

给你一个非负整数 x ，计算并返回 x 的 算术平方根 。

由于返回类型是整数，结果只保留 整数部分 ，小数部分将被 舍去 。

注意：不允许使用任何内置指数函数和算符，例如 pow(x, 0.5) 或者 x ** 0.5 。

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
x1 = 4
x2 = 9
ob1 = Solution()
print(ob1.mySqrt(x1))
print(ob1.mySqrt(x2))
```

### 牛頓法

```
def squareroot(input_num):
    root = input_num/2
    for k in range(20):
        root = (1/2)* (root + (input_num/root))
    return root

print(squareroot(3))
```

思路總結

1. 二分查找，分成左右區間。

2. 牛頓法

## LeetCode 66. Plus One 加一


You are given a large integer represented as an integer array digits, where each digits[i] is the $i^{th}$ digit of the integer. 

The digits are ordered from most significant to least significant in left-to-right order. 

The large integer does not contain any leading 0's.

Increment the large integer by one and return the resulting array of digits.

給定一個由 整數 組成的 非空 數組所表示的非負整數，在該數的基礎上加一。

最高位數字存放在數組的首位， 數組中每個元素只存儲單個數字。

你可以假設除了整數 0 之外，這個整數不會以零開頭。


Example 1:

```
Input: digits = [1,2,3]
Output: [1,2,4]
Explanation: The array represents the integer 123.
Incrementing by one gives 123 + 1 = 124.
Thus, the result should be [1,2,4].
```

Example 2:

```
Input: digits = [4,3,2,1]
Output: [4,3,2,2]
Explanation: The array represents the integer 4321.
Incrementing by one gives 4321 + 1 = 4322.
Thus, the result should be [4,3,2,2].
```

Example 3:

```
Input: digits = [9]
Output: [1,0]
Explanation: The array represents the integer 9.
Incrementing by one gives 9 + 1 = 10.
Thus, the result should be [1,0].
```

Constraints:

1. 1 <= digits.length <= 100

2. 0 <= digits[i] <= 9

3.  digits does not contain any leading 0's.

思路總結

1. 給一個陣列(数组, Array)，代表一個十進位數，陣列的 0 下標是十進位數的高位。要求計算這個十進位數加一以後的結果。

2. 從陣列尾部開始由後往前掃，逐位進位即可。最高位若還有進位的需要則在數組內第 0 位再插入一個 1。

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
x = [1,2,3]
ob1 = Solution1()
print(ob1.plusOne([1,2,3]))
class Solution2:
    def plusOne(self, digits: List[int]) -> List[int]:
        return list(map(int, list(str(int(''.join(map(str, digits))) + 1))))
x = [1,2,3]
ob2 = Solution2()
print(ob2.plusOne([1,2,3]))
```


## LeetCode 70. Climbing Stairs 爬楼梯

You are climbing a staircase. It takes n steps to reach the top.

Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

假设你正在爬楼梯。需要 n 阶你才能到达楼顶。

每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？

```
class Solution:
    def climbStairs(self, n):
        prev, current = 0, 1
        for i in range(n):
            prev, current = current, prev + current
        return current
x1 = 2
x2 = 3
ob1 = Solution()
print(ob1.climbStairs(x1))
print(ob1.climbStairs(x2))
```

思路總結

1. 動態規劃，遞迴公式 f(n-1) + f(n-2)，其結果就是費氏數列。來判斷該值有沒有在字典裡面。相對與第一種課堂範例來的理想。


