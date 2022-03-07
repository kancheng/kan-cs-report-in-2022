# AATCC - 算法分析和复杂性理论 - Analysis of Algorithms and Theory of Computational Complexity

> 2101212850 干皓丞

PKU 2022 個人實驗報告作業


## Details

針對算法分析和复杂性理论所做的針對性知識點筆記。

## Log

1. W1 : 

- LeetCode : `1. Two Sum 兩數之和` , `69. Sqrt(x) x 的平方根`, `70. Climbing Stairs 爬楼梯`.

- Knowledge Point : LeetCode 1, 69, 70.

2. W2 : 

- LeetCode : `7. Reverse Integer 整数反转`, `13. Roman to Integer 羅馬數字轉整數`, `66. Plus One 加一`.

- Knowledge Point : `平方根函數`, `＊牛頓法`,  - `乱序字符串检查 - 逐字检查`, `乱序字符串检查 - 排序与比较`, `＊乱序字符串检查 - 计数与比较(ASCII)`, `＊栈 - 十进制转二进制`, `栈 - 2 ~ 16 进制轉換`

3. W3 :


## 平方根函數

```
import math
a =100
print(math.sqrt(a))
```

## `*` 牛頓法


![](w2-kp-1.png)

```
def squareroot(input_num):
    root = input_num/2
    for k in range(20):
        root = (1/2)* (root + (input_num/root))
    return root

print(squareroot(3))
```


## 乱序字符串检查 - 逐字检查

## 乱序字符串检查 - 排序与比较

## `＊` 乱序字符串检查 - 计数与比较(ASCII)

## `＊`栈 - 十进制转二进制

## 栈 - 2 ~ 16 进制轉換

