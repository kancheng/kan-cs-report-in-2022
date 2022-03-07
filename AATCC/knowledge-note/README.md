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

乱序字符串是指一个字符串只是另一个字符串的重新排列。例如，'heart' 和 'earth' 就是乱序字符串。 'python' 和'typhon' 也是。为了简单起见，我们假设所讨论的两个字符串具有相等的长度，并且他们由26个小写 字母集合组成。我们的目标是写一个布尔函数，它将两个字符串做参数并返回它们是 不是乱序。

```
def anagramSolution1(s1, s2):
    alist = list(s2)
    pos1 = 0
    stillOK = True
    while pos1 < len(s1) and stillOK:
        pos2 = 0
        found = False
        while pos2 < len(alist) and not found:
            if s1[pos1] == alist[pos2]:
                found = True
            else:
                pos2 = pos2 + 1
        if found :
            alist[pos2] = None
            pos1 = pos1 + 1
        else:
            stillOK = False
    return stillOK and (len(list(filter(None, alist))) == 0)
print(anagramSolution1('eat', 'eat'))
print(anagramSolution1('eat', 'ade'))
```

## 乱序字符串检查 - 排序与比较

```
def anagramSolution2(s1, s2):
    alist1 = list(s1)
    alist2 = list(s2)
    alist1.sort()
    alist2.sort()
    pos = 0
    matches = True

    while pos < len(s1) and matches:
        if alist1[pos] == alist2[pos] :
            pos = pos + 1
        else:
            matches = False
    return matches

print(anagramSolution2('eat', 'eat'))
print(anagramSolution2('eat', 'ade'))
```

## `＊` 乱序字符串检查 - 计数与比较(ASCII)

![](w2-kp-2.png)

```
# 利用 ASCII TABLE

def anagramSolution3(s1, s2):
    c1 = [0] * 26
    c2 = [0] * 26
    for i in range(len(s1)):
        pos = ord(s1[i]) - ord('a')
        c1[pos] = c1[pos] + 1
    for i in range(len(s2)):
        pos = ord(s2[i]) - ord('a')
        c2[pos] = c2[pos] + 1
    j = 0
    stillOK = True
    while j < 26 and stillOK:
        if c1[j] == c2[j]:
            j = j + 1
        else :
            stillOK = False
    return stillOK

print(anagramSolution3('eat', 'eat'))
print(anagramSolution3('eat', 'ade'))
```

## `＊`栈 - 十进制转二进制

### 栈, Stack, 堆疊

栈(Last In First Out，LIFO)是一个项的有序集合，其 中添加移除新项总发生在同一端。这一端通常称为“顶部”。与顶部对应的端称为“底部”。

栈的抽象数据类型由以下结构和操作定义。栈被构造为项的有序集合，其中项被添加和从末端移除的位置称为“顶部”。

栈操作如下:

1. Stack() 创建一个空的新栈。 它不需要参数，并返回一个空栈。

2. push(item) 将一个新项添加到栈的顶部。它需要 item 做参数并不返回任何内容。

3. pop() 从栈中删除顶部项。它不需要参数并返回 item 。栈被修改。

4. peek() 从栈返回顶部项，但不会删除它。不需要参数。不修改栈。

5. isEmpty() 测试栈是否为空。不需要参数，并返回布尔值。

6. size() 返回栈中的 item 数量。不需要参数，并返回一个整数。

### 栈, Stack, 堆疊 - Python 實現

```
class Stack:
    def __init__(self):
        self.items = []
    def isEmpty(self):
        return self.items == []
    def push(self, item):
        self.items.append(item)
    def pop(self):
        return self.items.pop()
    def peek(self):
        return self.items[len(self.items) - 1]
    def size(self):
        return len(self.items)

s = Stack()
print(s.isEmpty())
s.push(4)
s.push('dog')
print(s.peek())
s.push(True)
print(s.size())
print(s.isEmpty())
s.push(8.4)
print(s.pop())
print(s.pop())
print(s.size())
```

### 栈, Stack, 堆疊 - 十进制转二进制 Python 實現

![](w2-kp-3.png)

```
class Stack:
    def __init__(self):
        self.items = []
    def isEmpty(self):
        return self.items == []
    def push(self, item):
        self.items.append(item)
    def pop(self):
        return self.items.pop()
    def peek(self):
        return self.items[len(self.items) - 1]
    def size(self):
        return len(self.items)

def divideBy2(decNumber):
    remstack = Stack()
    while decNumber > 0 :
        rem = decNumber % 2
        remstack.push(rem)
        decNumber = decNumber // 2
    binString = ""
    while not remstack.isEmpty():
        binString = binString + str(remstack.pop())
    return binString

print(divideBy2(100))
```

## 栈 - 2 ~ 16 进制轉換 Python 實現

```
class Stack:
    def __init__(self):
        self.items = []
    def isEmpty(self):
        return self.items == []
    def push(self, item):
        self.items.append(item)
    def pop(self):
        return self.items.pop()
    def peek(self):
        return self.items[len(self.items) - 1]
    def size(self):
        return len(self.items)

def baseConverter(decNumber, base):
    digits = "0123456789ABCDEF"
    remstack = Stack()
    while decNumber > 0:
        rem = decNumber % base
        remstack.push(rem)
        decNumber = decNumber // base
    newString = ""
    while not remstack.isEmpty():
        newString = newString + digits[remstack.pop()]
    return newString

print(baseConverter(109, 16))
```