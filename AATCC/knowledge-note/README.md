# AATCC - 算法分析和复杂性理论 - Analysis of Algorithms and Theory of Computational Complexity

> 2101212850 干皓丞

PKU 2022 個人實驗報告作業


## Knowledge Point

### Define

名詞對應 : 

https://github.com/kancheng/kan-cs-report-in-2022/blob/main/AATCC/define.md

### Log

課堂紀錄 :

https://github.com/kancheng/kan-cs-report-in-2022/blob/main/AATCC/log.md

1. 平方根函數

2. ＊牛頓法

3. 乱序字符串检查 - 逐字检查

4. 乱序字符串检查 - 排序与比较

5. ＊乱序字符串检查 - 计数与比较(ASCII)

6. ＊栈 - 十进制转二进制

7. 栈 - 2 ~ 16 进制轉換

8. 堆疊、链表應用

9. 链表, Linked List, 鏈結串列

10. 数组, Array, 陣列

11. 栈, Stack, 堆疊

12. 队列, Queue, 佇列

13. 查找, Search, 搜索, 搜尋 - 順序查找, 二分查找, Hash 查找

14. Map 抽象数据类型

15. 递归与分治

16. 排序 - 冒泡排序, 选择排序, 归并排序, 快速排序, 插入排序

17. 环形链表

18. 貪心法

- 貪心法 - 找最少硬币

- 貪心法 - LC 122. 买卖股票的最佳时机 II

- 貪心法 - LC 392. 判断子序列

- 貪心法 - LC 263. 丑数

19. 动态规划 Dynamic Programming, DP

- 动态规划 - 找最少硬币

- 动态规划 - LC 70. 爬楼梯

- 动态规划 - LC 62 & 63 不同路径 & 从一维到二维的扩展

20. 矩阵相乘加括号

21. 从一维到二维的扩展 with LC 746 & LC 120 說明

22. 矩阵相乘加括号 with LC 123 說明

23. 多起点多终点最短路径问题

24. 不满足优化子结构的例子 with LC 300 & LC 53

25. 最长公共子序列 (Longest Common Subsequence, LCS)

26. 背包问题 (Knapsack Problem)

27. 投资问题

28. 编辑距离, LC 72, LC 312

29. 樹

- 樹各名詞定義

- 定义树（Tree）

- Know Thy Complexities!

- Binary Tree & Binary Search Tree

- 树的表示-列表

- 树的表示-类

- 树和链表, LC 100, LC 112, LC 226

- 树的遍历(Traversal), LC 144

- 分析树 (Parse Tree)

- 构建分析树

- 树的遍历 & 分析树 (Parse Tree) LC 105

31. 基于二叉堆实现优先队列

- 优先队列 priority queue

- 二叉堆

- 二叉堆操作

- 二叉堆实现

- 构建二叉堆操作

- 二叉查找树

- 二叉查找树 LC 235

32. Trie & 实际问题

- Trie 树的基本结构、核心思想、基本性质、实现，LC 208 LC 720

33. LC 347 (桶排序)

34. 圖 & LC 997 (KP)

35. BFS and DFS LC 102、LC 104、LC 111.

36. LC 787. Cheapest Flights Within K Stops, K 站中转内最便宜的航班 and LC 934. Shortest Bridge 最短的桥

37. 最短路径问题

38. 狄克斯特拉（Dijkstra）算法

39. 贝尔曼-福特（Bellman-Ford）算法

40. 算法复杂度

41. 基本函数类

42. 递推方程与算法分析

43. 蒙地卡羅

44. 蒙地卡羅 & PI

45. 近似算法


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

是计算机科學中的一種抽象資料型別，只允許在有序的線性資料集合的一端（稱為堆疊頂端，英語：top）進行加入数据（英語：push）和移除数据（英語：pop）的運算。因而按照後進先出（LIFO, Last In First Out）的原理運作。常與另一種有序的線性資料集合佇列相提並論。堆疊常用一維数组或連結串列來實現。

栈(Last In First Out，LIFO)是一个项的有序集合，其中添加移除新项总发生在同一端。这一端通常称为“顶 部”。与顶部对应的端称为“底部”。


栈的抽象数据类型由以下结构和操作定义。栈被构造为项的有序集合，其中项被添加和从末端移除的位置称为“顶部”。

栈操作如下:

1. Stack() 创建一个空的新栈。 它不需要参数，并返回一个空栈。

2. push(item) 将一个新项添加到栈的顶部。它需要 item 做参数并不返回任何内容。

3. pop() 从栈中删除顶部项。它不需要参数并返回 item 。栈被修改。

4. peek() 从栈返回顶部项，但不会删除它。不需要参数。不修改栈。

5. isEmpty() 测试栈是否为空。不需要参数，并返回布尔值。

6. size() 返回栈中的 item 数量。不需要参数，并返回一个整数。

```
# 1. Stack() 创建一个空的新栈。 它不需要参数，并返回一个空栈。
class Stack:
    def __init__(self):
        self.items = []
# 5. isEmpty() 测试栈是否为空。不需要参数，并返回布尔值。
    def isEmpty(self):
        return self.items == []

    def push(self, item):
        self.items.append(item)
# 2. push(item) 将一个新项添加到栈的顶部。它需要 item 做参数并不返回任何内容。

    def pop(self):
        return self.items.pop()
# 3. pop() 从栈中删除顶部项。它不需要参数并返回 item 。栈被修改。

    def peek(self):
        return self.items[len(self.items) - 1]
# 4. peek() 从栈返回顶部项，但不会删除它。不需要参数。不修改栈。

    def size(self):
        return len(self.items)
# 6. size() 返回栈中的 item 数量。不需要参数，并返回一个整数。
```

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

## 队列, Queue, 佇列

計算機科學中的一種抽象資料型別，是先進先出（FIFO, First-In-First-Out）的線性表。在具體應用中通常用鍊表或者數組來實現。佇列只允許在後端（稱為rear）進行插入操作，在前端（稱為front）進行刪除操作。佇列的操作方式和堆疊類似，唯一的區別在於佇列只允許新數據在後端進行添加。

队列是项的有序结合，其中添加新项的一端称为队尾，移除项的一端称为队首。(First In First Out，FIFO)

队列操作如下:

- 1. Queue() 创建一个空的新队列。 它不需要参数，并返回一个空队列。

- 2. enqueue(item) 将新项添加到队尾。 它需要 item 作为参数，并不返回任何内容。 

- 3. dequeue() 从队首移除项。它不需要参数并返回 item。 队列被修改。

- 4. isEmpty() 查看队列是否为空。它不需要参数，并返回布尔值。

- 5. size() 返回队列中的项数。它不需要参数，并返回一个整数。

```
# 1. Queue() 创建一个空的新队列。 它不需要参数，并返回一个空队列。
class Queue:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return self.items == []
# 4. isEmpty() 查看队列是否为空。它不需要参数，并返回布尔值。

    def enqueue(self, item):
        self.items.insert(0, item)
# 2. enqueue(item) 将新项添加到队尾。 它需要 item 作为参数，并不返回任何内容。

    def dequeue(self):
        return self.items.pop()
# 3. dequeue() 从队首移除项。它不需要参数并返回 item。 队列被修改。

    def size(self):
        return len(self.items)
# 5. size() 返回队列中的项数。它不需要参数，并返回一个整数。
```

```
class Queue:
    def __init__(self):
        self.items = []
    def isEmpty(self):
        return self.items == []
    def enqueue(self, item):
        self.items.insert(0, item)
    def dequeue(self):
        return self.items.pop()
    def size(self):
        return len(self.items)
```

```
## e.g.1
def function(decNumber):
    remstack = Stack()
    while decNumber > 0:
        rem = decNumber % 2
        remstack.push(rem)
        decNumber = decNumber // 2
    binString = ""
    while not remstack.isEmpty():
        binString = binString + str(remstack.pop())
    return binString
print(function (10))
print(function (9))
print(function (8))
print(function (7))
## e.g.2
def function(input_num, n):
    root = input_num / 2
    for k in range(n):
        root = (1 / 2) * (root + (input_num / root))
    return root
print(function (10, 4))
print(function (9, 6))
print(function (8, 7))
print(function (7, 4))
## e.g.3
def function(namelist, num):
    simqueue = Queue()
    for name in namelist:
        simqueue.enqueue(name)
    while simqueue.size() > 1:
        for i in range(num):
            simqueue.enqueue(simqueue.dequeue())
        simqueue.dequeue()
    return simqueue.dequeue()
print(function ([4, 3, 6, 9, 14, 2, 5], 8))
```

## 呼應堆疊實作 LeetCode 20. Valid Parentheses 有效的括号

Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

An input string is valid if:

1. Open brackets must be closed by the same type of brackets.

2. Open brackets must be closed in the correct order.


给定一个只包括 '(', ')', '{', '}', '[' and ']' 的字符串 s ，判断字符串是否有效。

有效字符串需满足：

1. 左括号必须用相同类型的右括号闭合。

2. 左括号必须以正确的顺序闭合。


### 解题思路

- 遇到左括号就进栈push，遇到右括号并且栈顶为与之对应的左括号，就把栈顶元素出栈。最后看栈里面还有没有其他元素，如果为空，即匹配。

- 需要注意，空字符串是满足括号匹配的，即输出 true。

### 補充

括號匹配是使用棧解決的經典問題。題意其實就像我們在寫代碼的過程中，要求括號的順序是一樣的，有左括號，相應的位置必須要有右括號。

如果還記得編譯原理的話，編譯器在 詞法分析的過程中處理括號、花括號等這個符號的邏輯，也是使用了棧這種數據結構。

再舉個例子，linux 系統中，cd 這個進入目錄的命令我們應該再熟悉不過了。

```
cd a/b/c/../../
```

這個命令最後進入a目錄，系統是如何知道進入了a目錄呢 ，即為棧的應用


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
x = "()"
ob = Solution()
print(ob.isValid(x))
```

## `*` 用栈实现队列 - LeetCode 232. Implement Queue using Stacks

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
    
# Your MyQueue object will be instantiated and called as such:
# obj = MyQueue()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.peek()
# param_4 = obj.empty()
x = ["MyQueue","push","push","peek","pop","empty"]
obj = MyQueue()
print(obj.push(x))
param_2 = obj.pop()
param_3 = obj.peek()
param_4 = obj.empty()
print(param_2)
print(param_3)
print(param_4)
    
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

## 數組, Array, 陣列 VS List

![](w3-kp-1.png)

![](w3-kp-2.png)

```
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
        NewNode = ListNode(8)
        print(NewNode)

    head.next, NewNode.next = NewNode, head.next
    print(head)

    head, NewNode.next = NewNode, head
    print(head)
```

## 反转链表 - eetCode 206. Reverse Linked List

Given the head of a singly linked list, reverse the list, and return the reversed list.

给你单链表的头节点 head ，请你反转链表，并返回反转后的链表。


## 解题思路

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

## Reference

1. https://ithelp.ithome.com.tw/m/articles/10271920

2. https://ithelp.ithome.com.tw/articles/10263980

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

## 两两交换链表中的节点 - LeetCode 24. Swap Nodes in Pairs

Given a linked list, swap every two adjacent nodes and return its head. You must solve the problem without modifying the values in the list's nodes (i.e., only nodes themselves may be changed.)

给你一个链表，两两交换其中相邻的节点，并返回交换后链表的头节点。你必须在不修改节点内部的值的情况下完成本题（即，只能进行节点交换）。

## 解题思路

两两相邻的元素，翻转链表

`pre->a->b->b.next to pre->b->a->b.next`

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

## Reference

1. https://ithelp.ithome.com.tw/m/articles/10271920

2. https://ithelp.ithome.com.tw/articles/10263980

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

## LeetCode 141. Linked List Cycle 环形链表


Given head, the head of a linked list, determine if the linked list has a cycle in it.

There is a cycle in a linked list if there is some node in the list that can be reached again by continuously following the next pointer. Internally, pos is used to denote the index of the node that tail's next pointer is connected to. Note that pos is not passed as a parameter.

Return true if there is a cycle in the linked list. Otherwise, return false.


给你一个链表的头节点 head ，判断链表中是否有环。

如果链表中有某个节点，可以通过连续跟踪 next 指针再次到达，则链表中存在环。 为了表示给定链表中的环，评测系统内部使用整数 pos 来表示链表尾连接到链表中的位置（索引从 0 开始）。注意：pos 不作为参数进行传递。仅仅是为了标识链表的实际情况。

如果链表中存在环，则返回 true 。 否则，返回 false 

###  circular linked list

> 引用段落 : 自你所不知道的 C 語言: linked list 和非連續記憶體


環狀鏈結串列 (circular linked list) 是鏈結串列的最後一個節點所指向的下一個節點，會是第一個節點，而不像鏈結串列中的最後一個結點指向 NULL:
![](w4-kp-1.png)

其優點為:

從 head 找到 tail 的時間複雜度為 O(n)，但若新增一個 tail pointer (此為 last) 時間複雜度可降為 O(1)

- 容易做到反向查詢

- 若要走訪整個 linked list，任何節點都可作為起始節點

- 避免保留 NULL 這樣特別的記憶體地址 (在沒有 MMU 的 bare metal 環境中，(void `*`) 0 地址空間存取時，沒有特別的限制)


bare metal : https://en.wikipedia.org/wiki/Bare_machine

### 用「龜兔賽跑」(Floyd’s Cycle detection)來偵測是否有 cycle 產生。

Floyd’s Cycle detection : https://en.wikipedia.org/wiki/Cycle_detection

有 3 種狀態需要做討論

> * $a$ 為起始點
> * $b$ 為連接點
> * $c$ 為龜兔相遇位置

![](w4-kp-2.png)

我們需要求得 a, b, c 三點位置，才能進行處理。
假設 $\overline{ac}$ 距離為 $X$ ，這代表 tortoise 行經 $X$ 步，那麼 hare 走了 $2X$ 步，$X$ 數值為多少並不重要，只代表要花多少時間兩點才會相遇，不影響求出 $\mu$ 和 $\lambda$。

接下來要分成三個步驟來處理
1. tortoise 速度為每次一步，hare 為每次兩步，兩者同時從起點 $a$ 出發，相遇時可以得到點 $c$。若是上述「狀況 2: 頭尾相連」，在第 1 步結束就求完三點了
2. 兩點分別從點 $a$ 和 $c$ 出發，速度皆為一次一步，相遇時可得到點 $b$。因為 $\overline{ac}$ 長度為 $X$，那麼 $cycle$ $c$ 長度也為 $X$，相遇在點 $b$ 時，所走的距離剛好都是 $X - \overline{bc}$
3. 從點 $b$ 出發，速度為一次一步，再次回到點 $b$ 可得到 cycle 的長度

### cycle finding

如果只需要判斷是否為 circular linked list，那麼只要執行上述的第 1 部分。

除了計算 $\mu$ 和 $\lambda$，還需要記錄整個串列的長度，若不記錄，會影響到後續進行 sorting 一類的操作。

```cpp
static inline Node *move(Node *cur) { return cur ? cur->next : NULL; }

bool cycle_finding(Node *HEAD, Node **TAIL, int *length, int *mu, int *lambda) {
    // lambda is length
    // mu is the meet node's index
    Node *tortoise = move(HEAD);
    Node *hare = move(move(HEAD));

    // get meet point
    while (hare && tortoise) {    /* Maybe while (hare && tortoise && (hare != tortoise)) ?*/
        tortoise = move(tortoise);
        hare = move(move(hare));
    }

    // not loop
    if (!hare) {
        *TAIL = NULL;
        *length = 0;
        tortoise = HEAD;
        while (tortoise && (tortoise = move(tortoise)))
            (*length)++;
        return false;
    }

    // get mu
    *mu = 0;
    tortoise = HEAD;
    while (tortoise != hare) {
        (*mu)++;
        tortoise = tortoise->next;
        hare = hare->next;
    }

    // get lambda
    *lambda = 1;
    tortoise = move(tortoise);
    *TAIL = tortoise;
    while (tortoise != hare) {
        *TAIL = tortoise;
        (*lambda)++;
        tortoise = move(tortoise);
    }
    *length = *mu + *lambda;

    return true;
}
```

## LeetCode 15. 3Sum 三数之和

Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.

Notice that the solution set must not contain duplicate triplets.



给你一个整数数组 nums，有一个大小为 k 的滑动窗口从数组的最左侧移动到数组的最右侧。你只可以看到在滑动窗口内的 k 个数字。

滑动窗口每次只向右移动一位。返回 滑动窗口中的最大值 。


Example 1:

```
Input: nums = [-1,0,1,2,-1,-4]
Output: [[-1,-1,2],[-1,0,1]]
```

Example 2:

```
Input: nums = []
Output: []
```

Example 3:

```
Input: nums = [0]
Output: []
```

Constraints:

- 0 <= nums.length <= 3000
- -10^5 <= nums[i] <= 10^5


## 解题思路

用 map 提前计算好任意 2 个数字之和，保存起来，可以将时间复杂度降到 O(n^2)。这一题比较麻烦的一点在于，最后输出解的时候，要求输出不重复的解。数组中同一个数字可能出现多次，同一个数字也可能使用多次，但是最后输出解的时候，不能重复。例如 [-1，-1，2] 和 [2, -1, -1]、[-1, 2, -1] 这 3 个解是重复的，即使 -1 可能出现 100 次，每次使用的 -1 的数组下标都是不同的。

这里就需要去重和排序了。map 记录每个数字出现的次数，然后对 map 的 key 数组进行排序，最后在这个排序以后的数组里面扫，找到另外 2 个数字能和自己组成 0 的组合。

![](w4-kp-3.png)

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

## 查找, Search, 搜索, 搜尋

- 順序查找

- 二分查找

-  Hash 查找

# 13. Search 名詞釋疑

> 整理於台灣義守大學與維基百科


## 雜湊 (Hash), 哈希

雜湊是因為他的特性很適合來做加密的運算，但真的不等同於加密。

> 雜湊（英語：Hashing）是電腦科學中一種對資料的處理方法，通過某種特定的函式/演算法（稱為雜湊函式/演算法）將要檢索的項與用來檢索的索引（稱為雜湊，或者雜湊值）關聯起來，生成一種便於搜尋的資料結構（稱為雜湊表）。舊譯哈希（誤以為是人名而採用了音譯）。它也常用作一種資訊安全的實作方法，由一串資料中經過雜湊演算法（Hashing algorithms）計算出來的資料指紋（data fingerprint），經常用來識別檔案與資料是否有被竄改，以保證檔案與資料確實是由原創者所提供。
>
> 如今，雜湊演算法也被用來加密存在資料庫中的密碼（password）字串，由於雜湊演算法所計算出來的雜湊值（Hash Value）具有不可逆（無法逆向演算回原本的數值）的性質，因此可有效的保護密碼。

## 雜湊函數 (Hash function)

主要是將不定長度訊息的輸入，演算成固定長度雜湊值的輸出，且所計算出來的雜湊值必須符合兩個主要條件：

由雜湊值是無法反推出原來的訊息
雜湊值必須隨明文改變而改變。
舉例來說，雜湊函數就像一台果汁機，我們把蘋果香蕉你個芭樂 (資料) 都丟進去打一打、攪一攪，全部變得爛爛的很噁心對吧？！這時候出來的產物 (經過雜湊函數後的值)，是獨一無二的，沒有辦法反向組合成原來的水果 (資料)。倘若我們把蘋果改成紅龍果，出來的產物 (經過雜湊函數後的值) 就會跟著改變，變成桃紅色的，不再是原來的淡黃色。

承上述的例子，用紅龍果香蕉你個芭樂經過雜湊函數出來的顏色是桃紅色 (雜湊值)，那有沒有可能我用其他的水果也可以打出相同的顏色呢？但因為雜湊值的特性是無法反推的，所以如果真的打出相同的顏色的話，我們稱為碰撞 (Collision)。這就代表說這個雜湊值已經不安全，不再是獨一無二的了，需要更改雜湊函數。

## 雜湊表 (Hash table)

在用雜湊函數運算出來的雜湊值，根據 鍵 (key) 來儲存在數據結構中。而存放這些記錄的數組就稱為 雜湊表。


## 搜尋(Search)

搜尋就是在一堆資料中找出所要之特定資料。搜尋之主要核心動作為「比較」動作，必需透過比較才有辦法判斷是否尋找到特定資料。當資料量少時很容易，當資料量龐大時，如何快速搜尋為一重要課題。

一般電腦檔案都是一群結構記錄之集合(如上一單元之成績結構)。為了排序與搜尋，至少會設定其中一個欄位為資料之鍵值(key)。透過鍵值將資料排列順序，稱為排序。透過鍵值找到特定資料，稱為搜尋(search)。一般資料搜尋有下列分類：

## 依資料量大小

1. 內部搜尋：欲搜尋之資料較少，可直接載入記憶體中，進行搜尋動作。

2. 外部搜尋：欲搜尋之資料較多，無法一次載入記憶體進行搜尋動作。需使用外部輔助記憶體分批處理。

## 依搜尋時資料表格是否異動

1. 靜態搜尋：搜尋過程中，資料表格不會有任何異動(如：新增、刪除或更新)。例如：查閱紙本字典、電話簿。

2. 動態搜尋：搜尋過程中，資料表格會經常異動。

一般搜尋常見之演算法有，「循序搜尋」、「二分搜尋」、「二元樹搜尋」、「雜湊搜尋」。

## 循序搜尋法 (Sequential Search)

【定義】從第一個資料開始取出，依序一一與「目標資料」相互比較，直到找到所要元素或所有資料均尋找完為止，此方法稱「循序搜尋」。

【優點】(1) 程式容易撰寫。(2) 資料不須事先排序(Sorting)。

【缺點】 搜尋效率比較差(平均次數=(N+1)/2)，不管是否有排序，每次都必須要從頭到尾找一次。

【時間複雜度】

(1) 如果資料沒有重覆，找到資料就可終止，否則要找到資料結束。N筆資料，在最差之情況下，需作 N 次比較，O(N)。

(2) 在平均狀況下(假設資料出現與分佈之機率相等)需(N+1)/2次比較，所以平均時間與最差時間為O(N)，最好為O(1)=1次。

【演算法】

```c
int sequential_search(int list[], int n, int key) {
    int i;
    for (i = 0; i < n; i++){
        if (list[i] == key) return i+1;
        //比對陣列內的資料是否等於欲搜尋的條件
        //若找到符合條件的資料，就傳回其索引
    }
    return(-1);    
    //若找不到符合條件的資料，就傳回 -1
}
```
## 二分搜尋法 (Binary Search)

【定義】如果資料已先排序過，則可使用二分法來進行搜尋。二分法是將資料分成兩部份，再將鍵值與中間值比較，如鍵值相等則找到，小於再比前半段，大於再比後半段。如此，分段比較至找到或無資料為止。

【優點】搜尋效率佳(平均次數=Log2N)。

【缺點】 (1) 資料必需事先排序。(2) 檔案資料必需使是可直接存取或隨機檔。

【時間複雜度】因為每次比較都會比上一次少一半之資料，因此最多只需要比較。

【演算法】

```c
    Searchtime = 0;                   //搜尋次數初值設定為
    Middle = (int)((Low + High)/2);   //搜尋中間值
    do {
        Searchtime = Searchtime + 1;
        if (Temp[Middle] == Key)       //找到資料
        {
            printf("該數字是排在第 %d 個順位",Middle);
            //顯示資料位置
            printf("一共搜尋 %d 次",Searchtime);
            //顯示搜尋次數
            break;    //跳出迴圈
        }
        else if(Temp[Middle] < Key)
                Low = Middle + 1;          //改變左半部
            else  High = Middle - 1;     //改變右半部
        Middle = (int)((Low + High) / 2);  //改變中間值
    }
    while(Low <= High);
```

## 二元樹搜尋法 (Tree Search)

【定義】二元數是先將資料列建立為一棵二元搜尋樹，樹中每節點皆不小於左子樹(葉)，也不大於右子樹(葉)，也就是 左子樹的值≦樹根值≦右子樹的值。

【優點】 (1) 插入與刪除時，只需改變指標。(2) 二元樹效率較高(介於循序法與二分法間)。

【缺點】 (1) 有左、右兩指標，需較大記憶體空間。(2) 資料必須事先排序。

【時間複雜度】平均與最差時間為 O(N)

## 內插搜尋法(Interpolation Search)

【定義】內插搜尋法是二分搜尋法之改良版。是依照資料位置分佈，運用公式預測資料所在位置，再以二分法方式逼近。內插之預測公式為：

【優點】資料分佈平均時，搜尋速度極快。

【缺點】 (1) 需計算預測公式。(2) 資料必須事先排序。

【時間複雜度】取決於資料分部情形，平均而言優於 Log2N。

【演算法】
```c
int intsrch(int A[], int find) {
    int low, mid, high,Searchtime;
    low = 0;
    high = MAX - 1;
    Searchtime = 0;// 搜尋次數初值設定為
    while(low <= high) {
        mid = (high-low)* (find-A[low])/(A[high]-A[low])+ low;
        Searchtime = Searchtime + 1;   
        if(mid < low || mid > high)  return -1;
        if(find < A[mid])   high = mid - 1;
        else if(find > A[mid])
            low = mid + 1;
        else {
            printf("一共搜尋 %d 次, ",Searchtime);//顯示搜尋次數
            return mid;
        }
    }
    return -1;
}
```
## 雜湊搜尋法(Hashing Search)

存取資料時，並不依資料順序存取，是應用資料中某欄位之值代入事先設計好之函數(雜湊函數)，計算資料存放之位置。這種方式稱雜湊法(Hashing)。

【定義】將資料按照某特定法則轉換為資料儲存位置，應用時是以資料鍵值(key value)轉換。

【優點】 

(1) 搜尋速度最快。

(2) 資料不須是先排序。

(3) 在沒發生碰撞(collision)與溢位(overflow)之情況下，只需一次即可讀取。

(4) 搜尋速度與資料量大小無關。

(5) 保密性高，若不知雜湊函術，無法取得資料。

【缺點】 

(1) 浪費空間(因有溢位資料區)，並且儲存空間的利用率比循序檔差。

(2) 有碰撞問題，當資料檔記錄到一定量時會嚴重影響處理速度存取速度。

(3) 程式設計比較複雜。

(4) 大量資料無效率。

(5) 不適合循序型煤體，如磁帶。

【演算法】主要依雜湊函數之計算、碰撞與溢位為考量依據。以下簡單討論幾種雜湊函數與溢位處理方法。

## Reference

1. https://en.wikipedia.org/wiki/Search_algorithm

2. https://ithelp.ithome.com.tw/articles/10208884

3. https://en.wikipedia.org/wiki/Hash_function

4. https://zh.wikipedia.org/wiki/%E6%95%A3%E5%88%97%E5%87%BD%E6%95%B8


## 顺序查找, 循序搜尋法, Sequential Search

從第一個資料開始取出，依序與「目標資料」相互比較，直到找到所要元素或所有資料均尋找完為止，此方法稱「循序搜尋」。

```
# KP
def squentialSearch(alist, item):
    pos = 0
    found = False
    while pos < len(alist) and not found:
        if alist[pos] == item:
            found = True
        else :
            pos = pos + 1
    return found
testlist = [1, 2, 32, 8, 17, 19, 42, 13, 0]
print(squentialSearch(testlist, 3))
print(squentialSearch(testlist, 13))

# KP
def orderedSeqentialSearch(alist, item):
    pos = 0
    found = False
    stop = False
    while pos < len(alist) and not found and not stop:
        if alist[pos] == item:
            found = True
        else :
            if alist[pos] > item:
                stop = True
            else :
                pos = pos + 1
    return found
testlist = [0, 1, 2, 8, 13, 17, 19, 32, 42]
print(orderedSeqentialSearch(testlist, 3))
print(orderedSeqentialSearch(testlist, 13))
```

## 二分查找, 二分搜尋法, Binary Search

```
# KP
# 二分查找
def binarySearch(alist, item):
    first = 0
    last = len(alist) -1
    found = False
    while first <= last and not found:
        midpoint = (first + last) // 2
        if alist[midpoint] == item:
            found = True
        else:
            if item < alist[midpoint]:
                last = midpoint - 1
            else :
                first = midpoint + 1
    return found
testlist = [0, 1, 2, 8, 13, 17, 19, 32, 42]
print(binarySearch(testlist, 3))
print(binarySearch(testlist, 13))

# KP
# 二分查找
# 遞歸
def binarySearch(alist, item):
    if len(alist) == 0:
        return False
    else:
        midpoint = len(alist)//2
        if alist[midpoint] == item:
            return True
        else:
            if item < alist[midpoint]:
                return binarySearch(alist[:midpoint], item)
            else :
                return binarySearch(alist[midpoint + 1:], item)

testlist = [0, 1, 2, 8, 13, 17, 19, 32, 42]
print(binarySearch(testlist, 3))
print(binarySearch(testlist, 13))
```

## Hash 查找, 雜湊搜尋法, Hashing Search

存取資料時，並不依資料順序存取，是應用資料中某欄位之值代入事先設計好之函數(雜湊函數)，計算資料存放之位置。這種方式稱雜湊法(Hashing)。

- 簡單餘數法

- 分組求和法

- 平方取中法

![](w4-kp-4.png)

![](w4-kp-5.png)

![](w4-kp-6.png)


衝突 & List & 線性探測的開放尋址技術 e.g. [ 54, 26, 93, 17, 77, 31, 44, 55, 20]

## Map 抽象数据类型

- Map() 创建一个新的 map 。它返回一个空的 map 集合。

- put(key, val) 向 map 中添加一个新的键值对。如果键已经在 map 中，那么用新值替换旧值。

- get(key) 给定一个键，返回存储在 map 中的值或 None。

- del 使用 `del map[key]` 形式的语句从 map 中删除键值对。

- len() 返回存储在 map 中的键值对的数量。

- in 返回 True 对于 `key in map` 语句，如果给定的键在 map 中，否则为False。

最有用的 Python 集合之一是字典。

回想一下，字典是一种关联数据类型，你可以在其中存储键-值对。该键用于查找关联的值。我们经常将这个想法称为 `map`。

map 抽象数据类型定义如下。该结构是键与值之间的关联的无序集合。

map 中的键都是唯一的，因此键和值之间存在一对一的关系。

字典一个很大的好处是，给定一个键，我们可以非常快速地查找相关的值。

为了提供这种快速查找能力，我们需要一个支持高效搜索的实现。

我们可以使用具有顺序或二分查找的列表，但是使用如上所述的哈希表将更好，因为查找哈希表中的项可以接近 $O(1)$ 性能

```
class HashTable:
    def __init__(self):
        self.size = 11
        self.slots = [None] * self.size
        self.data = [None] * self.size
    def put(self, key, data):
        hashvalue = self.hashfunction(key, len(self.slots))
        if self.slots[hashvalue] == None:
            self.slots[hashvalue] = key
            self.data[hashvalue] = data
        else:
            if self.slots[hashvalue] == key:
                self.data[hashvalue] = data  # replace
            else:
                nextslot = self.rehash(hashvalue, len(self.slots))
                while self.slots[nextslot] != None and self.slots[nextslot] != key:
                    nextslot = self.rehash(nextslot, len(self.slots))
                if self.slots[nextslot] == None:
                    self.slots[nextslot] = key
                    self.data[nextslot] = data
                else:
                    self.data[nextslot] = data  # replace
    def hashfunction(self, key, size):
        return key % size
    def rehash(self, oldhash, size):
        return (oldhash + 1) % size
    def get(self, key):
        startslot = self.hashfunction(key, len(self.slots))
        data = None
        stop = False
        found = False
        position = startslot
        while self.slots[position] != None and not found and not stop:
            if self.slots[position] == key:
                found = True
                data = self.data[position]
            else:
                position = self.rehash(position, len(self.slots))
                if position == startslot:
                    stop = True
        return data
    def __getitem__(self, key):
        return self.get(key)
    def __setitem__(self, key, data):
        self.put(key, data)

H = HashTable()
H[54] = "cat"
H[26] = "dog"
H[93] = "lion"
H[17] = "tiger"
H[77] = "bird"
H[31] = "cow"
H[44] = "goat"
H[55] = "pig"
H[20] = "chicken"
print(H.slots)
print(H.data)
```

## 递归与分治

story = function (){
    从前有个山，
    山里有个庙，
    庙里有个和尚讲故事 story()
}
 
> 从前有个山，山里有个庙，庙里有个和尚讲故事，而故事是从前有个山，山里有个庙，庙里有个和尚讲故事

```c
def recursion(level, param1, param2, ...):
    # recursion terminator
    if level > MAX_LEVEL:
        print_result
        return
    # process logic in current level
    process_data(level, data ...)

    # drill down
    self.recursion(level + 1, p1, p2, ...)

    # reverse the current status if needed
    reverse_state(level)
```

### 递归 計算 n 的階乘 n!

$n! = 1 * 2 * 3 * ... * n$

```
def Factorial(n):
    if n <= 1:
        return 1
    return n * Factorial(n - 1)
```

### Recursion 压栈

```c
factorial(6)
6 * factorial(5)
6 * (5 * factorial(4))
6 * (5 * (4 * factorial(3)))
6 * (5 * (4 * (3 * factorial(2))))
6 * (5 * (4 * (3 *(2 *factorial(1)))))
6 * (5 * (4 * (3 *(2 *1 ))))
6 * (5 * (4 * (3 * 2)))
6 * (5 * (4 * 6))
6 * (5 * 24)
6 * 120
720
```
Fibonacci array: 1, 1, 2, 3, 4, 8, 13, 21, 34, …

$$ F(n) = F(n-1) + F(n-2) $$

```
def fib(n):
    if n == 0 or n == 1:
        return n
    return fib(n - 1) + fib (n - 2)
```

### 分治

```c
def divide_conquer(problem, param1, param2, ...):
    # recursion terminator
    if problem is None:
        print_result
        return
    # prepare data
    data = prepare_data(problem)
    subproblems = split_problem(problem, data)

    # conquer subproblems
    subresults1 = self.divide_conquer(subproblems[0], p1, ...)
    subresults2 = self.divide_conquer(subproblems[1], p1, ...)
    subresults3 = self.divide_conquer(subproblems[2], p1, ...)
    ...

    # process and generate the final result
    result = process_result(subresults1, subresults2, subresults3, ...)
```

### LeetCode 50. Pow(x, n) 遞迴與非遞迴

Implement pow(x, n), which calculates x raised to the power n (i.e., $x^n$).

实现 pow(x, n) ，即计算 x 的 n 次幂函数（即，$x^n$ ）。

解题思路

要求计算 Pow(x, n)

这一题用递归的方式，不断的将 n 2 分下去。注意 n 的正负数，n 的奇偶性。

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

## 排序

- 冒泡排序

- 选择排序

- 归并排序

- 快速排序

- 插入排序


- 冒泡排序, 氣泡排序法, Bubble Sort

又稱交換排序法，原理是從第一筆資料開始，逐一比較相鄰兩筆資料，如果兩筆大小順序有誤則做交換，反之則不動，接者再進行下一筆資料比較，所有資料比較完第1回合後，可以確保最後一筆資料是正確的位置。


- 选择排序, 選擇排序法, Selection Sort

原理是反覆從未排序數列中找出最小值，將它與左邊的數做交換。可以有兩種方式排序，一為由大到小排序時，將最小值放到末端;若由小到大排序時，則將最小值放到前端。例如:未排序的數列中找到最小值的資料，和第1筆資料交換位置，再從剩下未排序的資料列中找到最小值的資料，和第2筆資料交換位置，以此類推。


- 归并排序, 合併排序法, Merge Sort

原理是會先將原始資料分割成兩個資料列，接著再將兩個資料繼續分割成兩個資料列，依此類推，直到無法再分割，也就是每組都只剩下一筆資料時，再兩兩合併各組資料，合併時也會進行該組排序，每次排序都是比較最左邊的資料，將較小的資料加到新的資料列中，依此類推，直到最後合併成一個排序好的資料列為止。


- 快速排序, 快速排序法, Quick Sort

又稱分割交換排序法，是目前公認效率極佳的演算法，使用了分治法(Divide and Conquer)的概念。原理是先從原始資料列中找一個基準值(Pivot)，接著逐一將資料與基準值比較，小於基準值的資料放在左邊，大於基準值的資料放在右邊，再將兩邊區塊分別再找出基準值，重複前面的步驟，直到排序完為止。


- 插入排序, 插入排序法, Insertion Sort

原理是逐一將原始資料加入已排序好資料中，並逐一與已排序好的資料作比較，找到對的位置插入。例如:已有2筆排序好資料，將第3筆資料與前面已排序好的2筆資料作比較，找到對的位置插入，再將第4筆資料與前面已排序好的3筆資料作比較，找到對的位置插入，以此類推。

氣泡排序法 - Bubble Sort : https://ithelp.ithome.com.tw/articles/10276184

選擇排序法 - Selection Sort : https://ithelp.ithome.com.tw/articles/10276719

合併排序法 - Merge Sort : https://ithelp.ithome.com.tw/articles/10278179

快速排序法 - Quick Sort : https://ithelp.ithome.com.tw/articles/10278644

http://pages.di.unipi.it/marino/pythonads/SortSearch/TheQuickSort.html

插入排序法 - Insertion Sort : https://ithelp.ithome.com.tw/articles/10277360

### Python 相同結果，不同寫法

```
# method 1
temp = alist[i]
alistp[i] = alist[i + 1]
alist[i + 1] = temp

# method 2
alist[i], alist[i + 1] = alist[i + 1], alist[i]
```

```
# KP
# 冒泡排序
# code
def bubbleSort(alist):
    for passnum in range(len(alist) -1, 0, -1):
        for i in range(passnum):
            if alist[i] > alist[i + 1]:
                temp = alist[i]
                alist[i] = alist[i + 1]
                alist[i + 1] = temp
alist = [54, 26, 93, 17, 77, 31, 44, 55, 20]
bubbleSort(alist)
print(alist)

# Run Test
def bubbleSort(alist):
    for passnum in range(len(alist)-1, 0, -1):
        print(passnum)
        for i in range(passnum):
            if alist[i] > alist[i + 1]:
                temp = alist[i]
                alist[i] = alist[i+1]
                alist[i+1] = temp
            print(alist)

alist = [54, 26, 93, 17]
print("bubbleSort :")
print(alist)
bubbleSort(alist)
print(alist)
```

```
# KP
# 选择排序

def selectionSort(alist):
    for fillslot in range(len(alist) - 1, 0, -1):
        positionOfMax = 0
        for location in range(1, fillslot + 1):
            if alist[location] > alist[positionOfMax]:
                positionOfMax = location
        temp = alist[fillslot]
        alist[fillslot] = alist[positionOfMax]
        alist[positionOfMax] = temp
alist = [ 54, 26, 93, 17, 77, 31, 44, 55 ,20]
selectionSort(alist)
print(alist)

# Run Test
def selectionSort(alist):
    for fillsolt in range(len(alist)-1,0,-1):
        print(fillsolt)
        positionOfMax = 0
        for location in range(1, fillsolt + 1):
            if alist[location] > alist[positionOfMax]:
                positionOfMax = location
        temp = alist[fillsolt]
        alist[fillsolt] = alist[positionOfMax]
        alist[positionOfMax] = temp
        print(alist)

print("selectionSort : ")
alist = [54, 26, 93, 17]
print(alist)
selectionSort(alist)
print(alist)
```

```
# KP
# 归并排序
def mergeSort(alist):
    if len(alist) > 1:
        mid = len(alist)//2
        lefthalf = alist[:mid]
        righthalf = alist[mid:]
        mergeSort(lefthalf)
        mergeSort(righthalf)
        l, j, k = 0,0,0
        while i < len(lefthalf) and j < len (righthalf):
            if lefthalf[i] < righthalf[j]:
                alist[k] = lefthalf[i]
                i = i + 1
            else :
                alist[k] = righthalf[j]
                j = j + 1
            k = k + 1
        while i < len(lefthalf):
            alist[k] = lefthalf[i]
            i = i + 1
            k = k + 1
        while j < len(righthalf):
            alist[k] = righthalf[j]
            j = j + 1
            k = k + 1

k = 0
def mergeSort(alist):
    global k
    k = k + 1
    print("invoke function %d" %(k))
    print("Splitting ", alist)
    if len(alist) > 1:
        mid = len(alist)//2
        lefthalf = alist[:mid]
        righthalf = alist[mid:]
        mergeSort(lefthalf)
        mergeSort(righthalf)
        i, j, k = 0,0,0
        while i < len(lefthalf) and j < len(righthalf):
            if lefthalf[i] < righthalf[j]:
                alist[k] = lefthalf[i]
                i = i + 1
            else:
                alist[k] = righthalf[j]
                j = j + 1
            k = k + 1
        while i < len(lefthalf):
            alist[k] = lefthalf[i]
            i = i + 1
            k = k + 1
        while j < len(righthalf):
            alist[k] = righthalf[j]
            j = j + 1
            k = k + 1
    print("Merging ", alist)
print("mergeSort : ")
alist = [ 54, 26, 93, 17]
print(alist)
mergeSort(alist)
print(alist)
```

```
# KP
# 插入排序

def insertionSort(alist):
    for index in range(1, len(alist)):
        currentvalue = alist[index]
        position = index
        while position > 0 and alist[position - 1] > currentvalue:
            alist[position] = alist[position - 1]
            position = position - 1
        alist[position] = currentvalue
alist = [ 54, 26, 93, 17, 77, 31, 44, 55, 20]
insertionSort(alist)
print(alist)

def insertionSort(alist):
    for index in range(1, len(alist)):
        print(index)
        currentvalue = alist[index]
        position = index
        while position > 0 and alist[position - 1] > currentvalue:
            alist[position] = alist[position - 1]
            position = position - 1
        alist[position] = currentvalue
        print(alist)
print("insertionSort : ")
alist = [54, 26, 93, 17]
print(alist)
insertionSort(alist)
print(alist)
```

```
# KP
# 快速排序

def quickSort(alist):
    quickSortHelper(alist, 0, len(alist)-1)
def quickSortHelper(alist, first, last):
    if first < last:
        splitpoint = partition(alist, first, last)
        quickSortHelper(alist, first, splitpoint-1)
        quickSortHelper(alist, splitpoint+1, last)
def partition(alist,first,last):
    pivotvalue = alist[first]
    leftmark = first+1
    rightmark = last
    done = False
    while not done:
        while leftmark <= rightmark and alist[leftmark] <= pivotvalue:
            leftmark = leftmark + 1
        while alist[rightmark] >= pivotvalue and rightmark >= leftmark:
            rightmark = rightmark -1

        if rightmark < leftmark:
            done = True
        else:
            temp = alist[leftmark]
            alist[leftmark] = alist[rightmark]
            alist[rightmark] = temp
    temp = alist[first]
    alist[first] = alist[rightmark]
    alist[rightmark] = temp
    return rightmark


alist = [54,26,93,17,77,31,44,55,20]
print(alist)
quickSort(alist)
print(alist)
```

## 貪心法

### 貪心法 - 找最少硬币

贪心法，又称贪心算法、贪婪算法:在对问题求解时， 总是做出在当前看来是最好的选择。

简单地说，问题能够分解成子问题来解决，子问题的 最优解能递推到最终问题的最优解。这种子问题最优 解成为最优子结构。


![](w5-kp-1.png)

### 貪心法 - LC 122. 买卖股票的最佳时机 II

You are given an integer array prices where prices[i] is the price of a given stock on the $i^{th}$ day.

On each day, you may decide to buy and/or sell the stock. You can only hold at most one share of the stock at any time. However, you can buy it then immediately sell it on the same day.

Find and return the maximum profit you can achieve.


给定一个数组 prices ，其中 prices[i] 表示股票第 i 天的价格。

在每一天，你可能会决定购买和/或出售股票。你在任何时候最多只能持有 一股 股票。你也可以购买它，然后在 同一天 出售。
返回 你能获得的 最大 利润。

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

### 貪心法 - LC 392. 判断子序列

Given two strings s and t, return true if s is a subsequence of t, or false otherwise.

A subsequence of a string is a new string that is formed from the original string by deleting some (can be none) of the characters without disturbing the relative positions of the remaining characters. (i.e., "ace" is a subsequence of "abcde" while "aec" is not).

给定字符串 s 和 t ，判断 s 是否为 t 的子序列。

字符串的一个子序列是原始字符串删除一些（也可以不删除）字符而不改变剩余字符相对位置形成的新字符串。（例如，"ace"是"abcde"的一个子序列，而"aec"不是）。

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
if __name__ == '__main__':
    print(Solution().isSubsequence('dck', 'goodluck'))
```

### 貪心法 - LC 263. 丑数

An ugly number is a positive integer whose prime factors are limited to 2, 3, and 5.

Given an integer n, return true if n is an ugly number.

判断一个数字是否是“丑陋数字”，“丑陋数字”的定义是一个正数，并且因子只包含 2，3，5 。

```
class Solution:
    def isUgly(self, num):
        if num == 0:
            return False
        for i in [2,3,5]:
            while num % i == 0:
                num /= i
        return num == 1
if __name__ == "__main__":
    print(Solution().isUgly(18))
    print(Solution().isUgly(14))
```

## 动态规划 Dynamic Programming, DP

动态规划算法通常基于一个递推公式及一个或多个初始状态。当前子问题的解将由上一次子问题的解推出。

使用动态规划来解题只需要多项式时间复杂度，因此它比递归法、暴力法等要快许多。

状态 : 用来描述该问题的子问题的解。

状态转移方程 : 描述状态之间是如何转移的关系式。

DP 的基本原理 : 找到某个状态的最优解，然后在其帮助下，找到下一个状态的最优解。

1. 递归 + 记忆化→递推

2. 状态的定义: dp[n], dp[i][j], ...

3. 状态转移方程:dp[n] = best_of(dp[n-1], dp[n-2], ... ) 

4. 最优子结构

EX : (1) 找最少硬币 (2) 爬楼梯 (3) 走方格 (4) 从一维到二维的扩展 (5) 矩阵相乘加括号

### 动态规划 - 找最少硬币

以 26 分递归换硬币为例:

coinValueList = [1,5,10,25]

change = 26

$$
\text { numCoins }=\min \left\{\begin{array}{l}
1+\text { numCoins }(\text { originalamount }-1) \\
1+\text { numCoins }(\text { originalamount }-5) \\
1+\text { numCoins }(\text { originalamount }-10) \\
1+\text { numCoins }(\text { originalamount }-25)
\end{array}\right.
$$

![](w5-kp-2.png)

(1) 自上而下，递归求解

```
def recMC( coinValueList, change):
    minCoins = change
    if change in coinValueList:
        return 1
    else:
        for i in [c for c in coinValueList if c <= change]:
            numCoins = 1 + recMC(coinValueList, change - i)
            if numCoins < minCoins:
                minCoins = numCoins
    return minCoins
```

(2) 加入“备忘录”，去除冗余的递归求解

```
def recMC(coinValueList, change, knownResults):
    minCoins = change
    if change in coinValueList:
        knownResults[change] = 1
        return 1
    elif knownResults[change] > 0:
        return knownResults[change]
    else:
        for i in [c for c in coinValueList if c <= change]:
            numCoins = 1 + recDC(coinValueList, change - i, knownResults)
            if numCoins < minCoins:
                minCoins = numCoins
                knownResults[change] = minCoins
    return minCoins
```

(3) 自下而上，动态规划求解，状态转移方程

$$
\text { numCoins }=\min \left\{\begin{array}{l}
1+\text { numCoins }(\text { originalamount }-1) \\
1+\text { numCoins }(\text { originalamount }-5) \\
1+\text { numCoins }(\text { originalamount }-10) \\
1+\text { numCoins }(\text { originalamount }-25)
\end{array}\right.
$$

```
def dpMakeChange(coinValueList, change, minCoins):
    for cents in range(change + 1):
        coinCount = cents
        for j in [c for c in coinValueList if c <= cents]:
            if minCoins[cents - j] + 1 < coinCount:
                coinCount = minCoins[cents - j] + 1
        minCoins[cents] = coinCount
    return minCoins[change]
```

(4) 带自动找零功能，动态规划求解

```
def dpMakeChange(coinValueList, change, minCoins, coinsUsed):
    for cents in range(change + 1):
        coinCount = cents
        newCoin = 1
        for j in [c for c in coinValueList if c <= cents]:
            if minCoins[cents - j] + 1 < coinCount:
                coinCount = minCoins[cents - j] + 1
                newCoin = j
        minCoins[cents] = coinCount
        coinsUsed[cents] = newCoin
    return minCoins[change]
def printCoins(coinsUsed, change):
    coin = change
    while coin > 0:
        thisCoin = coinsUsed[coin]
        print(thisCoin)
        coin = coin - thisCoin
```

### 动态规划 - LC 70. 爬楼梯

You are climbing a staircase. It takes n steps to reach the top.

Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

假设你正在爬楼梯。需要 n 阶你才能到达楼顶。

每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？

爬樓梯，状态转移方程，空间节省策略

$$𝑑𝑝[𝑛] = 𝑑𝑝 [𝑛 − 1] + 𝑑𝑝 [𝑛 − 2]$$

```
# 70 爬樓梯 (KP)
class Solution:
    def climbStairs(self, n):
        prev, current = 0, 1
        for i in range(n):
            prev, current = current, prev + current
        return current
```

### 动态规划 - LC 62 & 63 不同路径 & 从一维到二维的扩展

There is a robot on an m x n grid. The robot is initially located at the top-left corner (i.e., grid[0][0]). The robot tries to move to the bottom-right corner (i.e., grid[m - 1][n - 1]). The robot can only move either down or right at any point in time.

Given the two integers m and n, return the number of possible unique paths that the robot can take to reach the bottom-right corner.

The test cases are generated so that the answer will be less than or equal to $2 * 10^9$

一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为 “Start” ）。

机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish” ）。

问总共有多少条不同的路径？

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

![](w5-kp-3.png)

A robot is located at the top-left corner of a m x n grid (marked 'Start' in the diagram below).

The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below).

Now consider if some obstacles are added to the grids. How many unique paths would there be?

An obstacle and space is marked as 1 and 0 respectively in the grid.

一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为 “Start” ）。

机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为 “Finish”）。

现在考虑网格中有障碍物。那么从左上角到右下角将会有多少条不同的路径？

网格中的障碍物和空位置分别用 1 和 0 来表示。

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

## 矩阵相乘加括号

1. 描述:

设 $A_1, A_2, ... , A_n$ 为矩阵序列，$A_i$ 为 $P_{i-1} \times P_{i}$ 阶矩阵，$i = 1,2,...,n$.

确定 乘法顺序使得元素相乘的总次数最少.

2. 输入:

向量 $P = <P_0, P_1, ... , P_n>$，n 个矩阵的行数、列数 实例:

$$P = <10, 100, 5, 50>$$

$$A_1: 10 \times 100, A_2: 100 \times 5, A_3: 5 \times 50$$

3. 括号位置不同，相乘总次数不同:

$$
(A_{1}A_{2})A_{3}: 10 \times 100 \times 5 + 10 \times 5 \times 50 = 7500 
A_{1}(A_{2}A_{3}): 10 \times 100 \times 50 + 100 \times 5 \times 50 = 75000
$$

4. 枚举算法:

加n个括号的方法有 $\frac{1}{n+1}\left(\begin{array}{c}2 n \\ n\end{array}\right)$ 是一个Catalan数，是指数级别:

搜索空间规模

$$
\begin{aligned}
W(n) &=\Omega\left(\frac{1}{n+1} \frac{(2 n) !}{n ! n !}\right)=\Omega\left(\frac{1}{n+1} \frac{\sqrt{2 \pi 2 n}\left(\frac{2 n}{e}\right)^{2 n}}{\sqrt{2 \pi n}\left(\frac{n}{e}\right)^{n \sqrt{2 \pi n}\left(\frac{n}{e}\right)^{n}}}\right) \\
&=\Omega\left(\frac{1}{n+1} \frac{n^{\frac{1}{2}} 2^{2 n} n^{2 n} e^{n} e^{n}}{e^{2 n} n^{\frac{1}{2}} n^{n} n^{\frac{1}{2}} n^{n}}\right)=\Omega\left(2^{2 n} / n^{\frac{3}{2}}\right)
\end{aligned}
$$

5. 确定子问题的边界:

输入 $P=< P_0, P_1, ..., P_n> , A_{i..j}$ 表示乘积 $A_{i}A_{i+1}...A{j}$ 的结果，其最后一次相乘是 $A_{i..j} = A_{i..k} A_{k+1..j}$

6. 确定优化函数和递推方程:

$m[i,j]$ 表示得到 $A_{i..j}$ 的最少的相乘次数，则递推方程和初值.

$$
m[i, j]= \begin{cases}0 & i=j \\ \min _{i \leq k<j}\left\{m[i, k]+m[k+1, j]+P_{i-1} P_{k} P_{j}\right\} & i<j\end{cases}
$$

输入 $P= <30, 35, 15, 5, 10, 20>, n=5$，矩阵链:$A_{1}A_{2}A_{3}A_{4}A{5}$，其中 $A_{1}$: $30 \times 35$，$A_{2}$: $35 \times 15$，$A_{3}$: $15 \times 5$，$A_{4}$: $5 \times 10$，$A_{5}$: $10 \times 20$

7. 备忘录:

| r | m[1,n] | m[2,n] | m[3,n] | m[4,n] | m[5,n] |
| - | - | - | - | - | - |
| r=1 | m[1,1]=0 | m[2,2]=0 | m[3,3]=0 | m[4,4]=0 | m[5,5]=0 |
| r=2 | m[1,2]=15750 | m[2,3]=2625 | m[3,4]=750 | m[4,5]=1000 |  |
| r=3 | m[1,3]=7875 | m[2,4]=4375 | m[3,5]=2500 |   |   |
| r=4 | m[1,4]=9375 | m[2,5]=7125 |   |   |   |
| r=5 | m[1,5]=11875 |   |   |   |   |

8. 解: 

$$(A_{1} (A_{2} A_{3})) (A_{4}A_{5})$$

```
class Matrix:
    def __init__(self, row_num=0, col_num=0, matrix=None):
        if matrix != None:
            self.row_num = len(matrix)
            self.col_num = len(matrix[0])
        else:
            self.row_num = row_num
            self.col_num = col_num
        self.matrix = matrix

def matrix_chain(matrixs):
    matrix_num = len(matrixs)
    m = [[0 for j in range(matrix_num)] for i in range(matrix_num)]
    for interval in range(1, matrix_num + 1): 
        for i in range(matrix_num - interval):
            j = i + interval
            m[i][j] = m[i][i] + m[i + 1][j] + matrixs[i].row_num * matrixs[i + 1].row_num * matrixs[j].col_num
            for k in range(i + 1, j):
                temp = m[i][k] + m[k + 1][j] + matrixs[i].row_num * matrixs[k + 1].row_num * matrixs[j].col_num
                if temp < m[i][j]:
                    m[i][j] = temp 
    return m[0][matrix_num - 1]

# Test
matrixs = [Matrix(30, 35), Matrix(35, 15), Matrix(15, 5), Matrix(5, 10), Matrix(10, 20)]
# print(matrixs)
result = matrix_chain(matrixs)
print(result)
```

## 从一维到二维的扩展 with LC 746 & LC 120 說明

1. LC 746. Min Cost Climbing Stairs

You are given an integer array cost where cost[i] is the cost of ith step on a staircase. Once you pay the cost, you can either climb one or two steps.

You can either start from the step with index 0, or the step with index 1.

Return the minimum cost to reach the top of the floor.

给你一个整数数组 cost ，其中 cost[i] 是从楼梯第 i 个台阶向上爬需要支付的费用。一旦你支付此费用，即可选择向上爬一个或者两个台阶。

你可以选择从下标为 0 或下标为 1 的台阶开始爬楼梯。

请你计算并返回达到楼梯顶部的最低花费。

```
class Solution:
    def minCostClimbingStairs(self, cost):
        cost.append(0)
        for i in range(2, len(cost)):
            cost[i] += min(cost[i - 1], cost[i - 2])
        return cost[-1]
```

![](w6-kp-6.png)

2. 动态规划

- (1). 状态定义: dp[i]表示到达第i级台阶最小花费

- (2). 初始化: dp[0] = cost[0]; dp[1]=cost[1]

- (3). 转移方程: dp[i] = min(dp[i-1], dp[i-2]) + cost[i] (i >= 2)

可直接在 cost 列表空间上 DP

转移方程改为: cost[i] += min(cost[i-1], cost[i-2]) (i >= 2)

也為 O(n) time, O(1) space

![](w6-kp-7.png)

3. LC 120

Given a triangle array, return the minimum path sum from top to bottom.

For each step, you may move to an adjacent number of the row below. More formally, if you are on index i on the current row, you may move to either index i or index i + 1 on the next row.

给定一个三角形 triangle ，找出自顶向下的最小路径和。

每一步只能移动到下一行中相邻的结点上。相邻的结点 在这里指的是 下标 与 上一层结点下标 相同或者等于 上一层结点下标 + 1 的两个结点。也就是说，如果正位于当前行的下标 i ，那么下一步可以移动到下一行的下标 i 或 i + 1 。

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

### (1) 定义状态函数:

dp[i][j] 表示( i,j) 位置的点到最低端的最小路径值。

### (2) 状态转移方程:

dp[i][j] = min {dp[i + 1][j], dp[i + 1][j + 1]} + triangle[i][j]


```{python}
import copy 
class Solution:
    def minimumTotal1(self, triangle):
        if not triangle or triangle == [[]]: return 0
        dp = copy.deepcopy(triangle)
        for items in range(len(dp) - 2, -1, -1):
            for idx in range(len(dp[items])):
                dp[items][idx] = min(dp[items + 1][idx], dp[items + 1][idx + 1]) + triangle[items][idx]
        # print(dp)
        # return dp[0][0] -> 引入二维列表
        return dp[0][0]
```

### (3) 引入一维列表

res = triangle[-1] -> 引入一维列表

```
def minimumTotal2(self, triangle):
    if not triangle or triangle == [[]]: return 0
    # res = triangle[-1] -> 引入一维列表
    res = triangle[-1]
    for items in range(len(triangle) - 2, -1, -1):
        for idx in range(len(triangle[items])):
            res[idx] = min(res[idx], res[idx+1]) + triangle[items][idx]
    return res[0]
```

### (4) 无需引入其他变量

return triangle[0][0] -> 无需引入其他变量

```
def minimumTotal3(self, triangle):
    if not triangle or triangle == [[]]: return 0
    for items in range(len(triangle) - 2, -1, -1): 
        for idx in range(len(triangle[items])):
            triangle[items][idx] = min(triangle[items + 1][idx], triangle[items + 1][idx + 1]) + triangle[items][idx] 
    # return triangle[0][0] -> 无需引入其他变量
    return triangle[0][0]
```

## 矩阵相乘加括号 with LC 123 說明

1. LC 123. Best Time to Buy and Sell Stock III 买卖股票的最佳时机 III

You are given an array prices where prices[i] is the price of a given stock on the ith day.

Find the maximum profit you can achieve. You may complete at most two transactions.

Note: You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).

给定一个数组，它的第 i 个元素是一支给定的股票在第 i 天的价格。

设计一个算法来计算你所能获取的最大利润。你最多可以完成两笔交易。

注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

### (1) 定义状态:

dp[i][j][k] 

i 天结束时的最高利润 = [天数][是否持有股票][卖出次数] 

i: 0, ..., n

j: 0, 1

k: 0, 1, 2

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

## 多起点多终点最短路径问题

![](w6-kp-8.png)

边上的数字代表路径的距离，任意起点 $S_{i}$ 到任意终点 $T_{k}$ 的所有路径最短距离是多少?

![](w6-kp-9.png)

存在两条最短路径，距离都是10。

![](w6-kp-11.png)

定义状态: F(V) 表示点到终点最短的距离。

状态转移方程:
    
$$
\begin{aligned}
&F\left(C_{l}\right)=\min _{m}\left\{C_{l} T_{m}\right\} \\
&F\left(B_{k}\right)=\min _{l}\left\{B_{k} C_{l}+F\left(C_{l}\right)\right\} \\
&F\left(A_{j}\right)=\min _{k}\left\{A_{j} B_{k}+F\left(B_{k}\right)\right\} \\
&F\left(S_{i}\right)=\min _{j}\left\{S_{i} A_{j}+F\left(A_{j}\right)\right\}
\end{aligned}
$$

- 优化函数的特点: 任何最短路径的子路径都是相对于子路径始点和终点的最短路径

- 求解步骤: 确定子问题的边界、从最小的子问题开始进行多步判断

![](w6-kp-10.png)


$$
\min _{i}\left\{F\left(\boldsymbol{S}_{\boldsymbol{i}}\right)\right\}=\mathbf{1 0}
$$

## 不满足优化子结构的例子 with LC 300 & LC 53

1. 說明

优化原则:

一个最优决策序列的任何子序列本身一定是相对于子序列的初始和结束状态的最优的决策序列。

例 : 求总长模 10 的最小路径

![](w6-kp-12.png)

最优解: 下、下、下、下

动态规划算法的解: 下、上、上、上

不满足优化原则，不能使用动态规划设计技术

2. LC 300

Given an integer array nums, return the length of the longest strictly increasing subsequence.

A subsequence is a sequence that can be derived from an array by deleting some or no elements without changing the order of the remaining elements. For example, [3,6,2,7] is a subsequence of the array [0,3,1,6,2,2,7].


给你一个整数数组 nums ，找到其中最长严格递增子序列的长度。

子序列是由数组派生而来的序列，删除（或不删除）数组中的元素而不改变其余元素的顺序。例如，[3,6,2,7] 是数组 [0,3,1,6,2,2,7] 的子序列。

定义状态函数 :

dp[i] 表示包含第 i 个元素的最大上升子序列的长度。

$$
\begin{gathered}
\operatorname{nums}=[10,9,2,5,3,7,101,18] \\
d p=[1,1,1,2,2,3,4,4] \\
\end{gathered}
$$

状态转移方程 :

$$
\begin{gathered}
d p[i]=\max _{0 \leq j<i}\{d p[j]+1,1\}, \text { if nums }[i]>\text { nums }[j] \\
\end{gathered}
$$

最终结果 :

$$
\begin{gathered}
\max _{i}\{d p[i]\}
\end{gathered}
$$

![](w6-kp-13.png)

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

3. LC 53

Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.

A subarray is a contiguous part of an array.

给你一个整数数组 nums ，请你找出一个具有最大和的连续子数组（子数组最少包含一个元素），返回其最大和。

子数组 是数组中的一个连续部分。

```
class Solution(object):
    def maxSubArray(self, nums):
        for i in range(1, len(nums)):
            nums[i]= nums[i] + max(nums[i-1], 0)
        return max(nums)
```

LC 53 概念說明

![](w6-kp-14.png)

![](w6-kp-15.png)

![](w6-kp-16.png)

```
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

## 最长公共子序列 (Longest Common Subsequence, LCS)

![](w6-kp-17.png)

![](w6-kp-18.png)

![](w6-kp-19.png)

![](w6-kp-20.png)

![](w6-kp-21.png)

![](w6-kp-22.png)


## 背包问题 (Knapsack Problem)

>
> 一个旅行者随身携带一个背包，可以放入背包的物品有 n 种，每种物品的重量和价值分别是 $w_{i}$ , $v_j$, $i = 1$, ... , n 。
>
> 如果背包的最大容量限制是 b，怎样选择放入背包的物品以使得背包的价值最大 ?
>


0/1 背包是動態規劃研究的重要問題，因為它提供了許多有用的見解。

語句：給定一組從 1 到 n 編號的 n 個物品，每個物品都有一個重量 wi 和一個值 vi，以及最大重量容量 W，最大化背包中物品的值的總和，使得重量小於或等於背包的容量。

天真的解決方案：

讓我們看看天真的解決方案 - 每個項目只有 2 個選擇，要么包含在背包中，要么忽略該項目。
如果包含項目，則通過減少容量 W - vi 並累積項目值來檢查剩餘項目 (N - 1)。否則，在容量和價值不變的情況下檢查剩餘項目 (N - 1)。同樣，下一個項目將有兩個選擇。如果您將其可視化為樹，它將類似於下面的決策樹：

![](w6-kp-1.png)

每個級別 d 有 $2^d$ 個選項，有 N 個項目，因此復雜度為 $2^N$。

另一種將每個項目視為位的方法，然後我們檢查設置和取消設置的所有可能組合，並找到在滿足權重約束時獲得的最大值。很明顯，我們需要檢查 (1 << n) 或 $2^N$ 次迭代。所以，天真的解決方案是 $2^N$。


表格法：

考慮一個非常簡單的例子 - 權重 = {1, 2, 3} 和值 ={6, 10, 12}，我們有容量為 5 的背包。

現在，讓我們使用表格方法實現相同的功能

![](w6-kp-2.png)

在列上，我們將容量從 0 增加到 W，即最大容量從 0 增加到 5。在每一行上，我們考慮項目，我們注意到它的權重和值。對於每一行，我們只考慮前幾行中考慮的項目，對於每一列，我們考慮那麼多容量。基本情況是重量為 0（無物品），無論容量如何，值都是 0，同樣，如果容量為 0，那麼我們不能放置任何物品，因此值將為 0。

第一行（權重為 1 的行）很簡單，我們的權重為 1，因此我們可以從容量 1 填充它的值。因為只有整行才會有值 6

對於第二行，現在權重為 2，我們可以將與其上方行相同的值填充到容量 2。對於容量 2，它將是 2 選擇 - 包括或排除當前項目。如果我們排除當前項目，則值將與最上面的第 6 行相同。如果我們包括，則值將是 = 當前值 (10) + d(1，當前容量 (2) - 重量(2)) = 10 + d( 1, 2- 2) = 10 + 0 = 10。最大值為 10，因此結果為 10。現在，d 函數是前一項 (1)，零權重 = 0。
這就是我們得到公式的方式：

```
d(i, w) = Math.Max( d(i - 1, w), d(i - 1, w - weight[i]) + value[i])
```


考慮第 3 行和容量 4，不包括第 3 項，我們從上面的行得到 16 個值，包括它我們發現值 = 12 + d(2, 1) = 12 + 6 = 18。

很明顯，我們對每個容量 0 到 W 和每個項目 0 到 N 只計算一次，所以復雜度是 O(NW)。

參考代碼：

```
// given N, maxWeight, weights and values
long[,] d = new long[N + 1, maxWeight + 1];

for (long i = 0; i < N; i++)
{
    for (long w = 0; w <= maxWeight; w++)
    {                    
        if (weights[i] <= w)
        {
            // Exclude or include
            d[i + 1, w] = Math.Max(d[i, w], d[i, w - weights[i]] + values[i]);
        }

        else
        {
            // Exclude
            d[i + 1, w] = d[i, w];
        }                    
    }
}
```

等等，我們如何將時間複雜度從 O(2 ^ N) 提高到 O(N * W)？

這是因為我們重用了已經計算好的解決方案。例如，如果容量 = 7，而不是嘗試不同的項目組合，如 4 + 3、2 + 5、1 + 6、2 + 4 等。我們只做一個計算來排除或包含當前項目。當我們包含當前項目時，我們正在重用已發現容量減少和項目更少的解決方案。

許多動態規劃問題遵循類似的模式，例如

- 1. 我們有優化功能 - 最大化價值，最小化距離等

- 2. 最優子結構 - 遞歸地找到子問題的最優解

- 3. 重疊子問題 - 相同的子問題一次又一次地解決。

動態編程解決每個子問題一次並重用結果。有兩種方法：

1. 自上而下：在子問題的遞歸計算過程中，我們存儲結果，所以當我們再次嘗試子問題時，我們直接使用存儲的結果而不是重新計算。因此，結果應該以可以在 O(1) 時間內檢索到的方式存儲 - 就像使用數組/字典一樣。

2. 自下而上：這裡我們嘗試解決較小的子問題，例如上面的項目和容量，然後到達更大的問題。更大的子問題的解決方案是通過使用已經計算的子問題的解決方案來生成的。

無論哪種情況，我們都需要找出子問題建立的狀態。例如，考慮的項目和剩餘容量是我們的狀態，無論剩餘的項目數量和相同 i 和 w 的總容量如何，我們都具有相同的值。識別狀態對於動態規劃至關重要。


```
from typing import List

def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        bagSize = len(s)+1
        itemSize = len(wordDict)
        dp = [False] * bagSize
        dp[0] = True
        # 排列而不是组合。 外遍历中的背包。
        # 允许重叠物品，背包溯源应从小值开始。
        # 当我们同时遇到一个 True 时中断
        # Permutation instead of combination. knapsack in outer traversal。
        # Allowed overlapping items，tracersal of knapsack should start with small value。
        # break when we meet a True at once
        for j in range(1, bagSize):
            for i in range(itemSize):
                if j-len(wordDict[i])>=0 and dp[j-len(wordDict[i])] and wordDict[i]==s[j-len(wordDict[i]):j]:                    
                    dp[j] = True
                    break
        return dp[-1]
```


## 投资问题


>
> 设有 m 元钱, n 项投资， 函数 $f_{i}(x)$ 表示将 x 元钱投入到第 i 项项目所产生的效益， i=1 ,... , n。
>
> 问:如何分配这 m 元钱，使得投资的总效益最高?
>

![](w6-kp-3.png)

1. 暴力求解

算法思想为对所有项目进行循环，通过限定条件：总投资金额 = y，得到所有符合的答案，从中选取最大值，即为所求。

```
if __name__ == '__main__':
    profitMatrix=[[0,11,12,13,14,15],
                  [0,0,5,10,15,20],
                  [0,2,10,30,32,40],
                  [0,20,21,22,23,24]]
    a=[0,0,0,0] #存放最优投资方案
    maxProfit=0
    sumMoney=0
    for x1 in range(6):
        for x2 in range(6):
            for x3 in range(6):
                for x4 in range(6):
                    x=x1+x2+x3+x4
                    if x==5:
                        sumMoney=profitMatrix[0][x1]+profitMatrix[1][x2]+profitMatrix[2][x3]+profitMatrix[3][x4]
                    if sumMoney>maxProfit:
                        maxProfit=sumMoney
                        a[0]=x1
                        a[1]=x2
                        a[2]=x3
                        a[3]=x4
    print("最大利润为："+str(maxProfit))
    print("最优投资方案为："+str(a))
```

2. 动态规划

算法思想為假设第 x 个项目投资 m 万元，则将 x 个项目的 y 万元投资问题分解为前 x-1 个项目投资 y-m 万元和第 x 个项目投资 m 万元。这样就可以将问题规模减小，直至仅有一个项目，此时的最佳投资方案就是其本身。

令 outspace[x][y] 表示前 x 个项目投资 y 万元得到的最大利润，令

profitMatrix[i][m]=fi(m)，则动态方程为：

outspace[x][y]=profitMatrix[x][m]+outspace[x-1][y-m]

限定边界为 outspace[0][k]=profitMatrix[0][k]

```
def getProfit(profitMatrix,outspace,maxProfit):
    for i in range(6):
        outspace[0][i]=profitMatrix[0][i]
    for i in range(1,4):
        for j in range(6):
            for m in range(j+1):
                if outspace[i][j]<profitMatrix[i][m]+outspace[i-1][j-m]:
                    outspace[i][j]=profitMatrix[i][m]+outspace[i-1][j-m]
                if maxProfit<outspace[i][j]:
                    maxProfit=outspace[i][j]
    return maxProfit
                
if __name__ =='__main__':
    profitMatrix=[[0,11,12,13,14,15],
                  [0,0,5,10,15,20],
                  [0,2,10,30,32,40],
                  [0,20,21,22,23,24]]
    outspace=[]
    for i in range(4):
        outspace.append([])
        for j in range(6):
            outspace[i].append(0)
    maxProfit=0
    a=getProfit(profitMatrix,outspace,maxProfit)
    print("最大利润为："+str(a))
```

## 背包问题 (Knapsack Problem)

# 15. 背包问题 (Knapsack Problem)

一个旅行者随身携带一个背包，可以放入背包的物品有 $n$ 种，每种物品的重量和价值分别是 $w_{j}, v_{j}, i = 1, .., n $。如果背包的最大容量限制是 $b$，怎样选择放入背包的物品以使得背包的价值最大 ?

## 背包问题是什么?

定义:给定一组物品,每种物品都有自己的重量和价格,在限定的总重量内,我们如何选择,才能使得物品的总价格最高;

背包问题的一个例子:应该选择哪些盒子,才能使价格尽可能地大,而保持重量小于或等于 15kg?

1. 0-1 背包问题;
2. 有界背包问题 (多重背包问题);
3. 无界背包问题 (完全背包问题);

1. 当每种物体都只有一个时, 上述描述为 0-1 背包问题;
2. 当每种物体都有多个时, 上述描述为有界背包问题;
3. 当每种物体都有无限个时, 上述描述为无界背包问题; 

(1) 01背包问题

- 二维动态规划

- 一维动态规划

(2) 完全背包问题

- 二维动态规划

- 一维动态规划

- 一维动态规划 + 省略取物品次数 k 操作

## 1) 01包问题

### 二维动态规划

若每種物品只能使用一次，則該問題為 01 背包問題。

定义状态 : dp[i][j]，代表前个物品，存入容量为的背包里的最大价值。

状态转移 : dp[ilil= max(dp[i-1][j],dp[i-1][j - w[i]] + v[i])，其中不取第个物品对应的项是 dp[i][j] = dp[i-1][j]，取第 i 个物品对应的是 dp[i][j] = dp[i-1][j-w[i]] + v[i]。

### 一维动态规划

从状态转移公式中可以看出来当前层的状态只与上一层状态有关，因此可以优化 dp 二维列表为一维列表，要注意的是要从后往前更新确保更新第 i 层用的是第 i-1 层的状态值。

## 2)完全背包问题

### 二维动态规划

完全背包问题与 01 背包问题最大的区别就是每一个物品可以选无数次，因此当我们考虑到第个物品时，我们应该考虑的情况是: 不选这个物品、选一次这个物品、选两次这个物品。到选的物品重量超过背包容量(k*w[i]>j)，然后再在这些情况中选价值最大的。

定义状态: dp[i][j]，代表前个物品，存入容量为的背包里的最大价值。

状态转移: dp[i][j] = max(dp[i-1][j], dp[i-1][j- k * w[i] + k * v[i])。

### 一维动态规划

与 01 背包问题同理,由于第 i 层状态只与第 1 层状态有关,因此从后往前更新确保更新第 i 层用的是第 i- 1 层的状态值。

一维动态规划 + 省略取物品次数 k 操作

在完全背包问题中,二维 dp 中求解 dp[i][j] 时用到它的上一格dp[i-1][j] 和 dp[i][j-w[i]]，它与 01 背包的区别就在于 dp[i][j-w[i]]是当前行的内容，在 dp[i][j] 之前它需要先被修改，所以在一维 dp 中 dp[j] 的求解顺序是从左往右。




## 0-1 背包问题 – 回溯法

每种物体被限制为只能选一次;

复杂度分析: $O(2^{n})$

## 0-1 背包问题 – 动态规划法:

复杂度分析: $O(nb)$

每种物体被限制为只能选一次;

首先定义动态规划的 dp 数组含义, 将 dp_{\{i,j\}} 定义为 :

在 0-i 的索引范围中挑选物体, 在最大负重为 j 时, 背包内物体的最大价值;

1) i == 0 时, 无论背包的负重如何都最大只能放入第 0 个物体

$d p^{\{0, j\}}=v^{\{0\}}, j=w^{\{0\}}, \ldots, b ;$

2) i > 0 时, 可知状态转移方程

$d p^{\{i, j\}}=\max \left(d p^{\{i-1, j\}}, d p^{\{i-1, j-w[i]\}}+v[i]\right)$

$i=1, \ldots, n-1 ; j=1, \ldots, b$


## 有界背包问题 – 转成 0-1 背包:

每种物体选择次数有限;

复杂度分析: $O\left(\sum n b\right)$

首先定义动态规划的 dp 数组含义，将 $d p^{\{i, j\}}$ 定义为:

在 0-i 的索引范围中挑选物体, 在最大负重为 j 时, 背包内物体的最大价值;

第 i 种物体有 $N_{i}$ 个, 那么此时总共有 $\boldsymbol{n}^{\prime}=\sum_{i}^{n} \boldsymbol{N}_{\boldsymbol{i}}$ 个物体，再根据 0-1 背包的方法来求解:

1) i == 0 时, 无论背包的负重如何都最大只能放入第 0 个物体,

$\boldsymbol{d} \boldsymbol{p}^{\{\mathbf{0}, j\}}=\boldsymbol{v}^{\{\mathbf{0}\}}, \boldsymbol{j}=\boldsymbol{w}^{\{\mathbf{0}\}}, \ldots, \boldsymbol{n}^{\prime} ;$

2) i > 0 时, 可知状态转移方程:

$\boldsymbol{d} \boldsymbol{p}^{\{\boldsymbol{i}, \boldsymbol{j}\}=} \boldsymbol{\operatorname { m a x }}\left(\boldsymbol{d} \boldsymbol{p}^{\{\boldsymbol{i}-\mathbf{1}, \boldsymbol{j}\}}, \boldsymbol{d} \boldsymbol{p}^{\{\boldsymbol{i}-\mathbf{1}, \boldsymbol{j}-w[i]\}}+\boldsymbol{v}[\boldsymbol{i}]\right)$
$\boldsymbol{i}=\mathbf{1}, \ldots, \boldsymbol{n}-\mathbf{1}^{\prime} ; \boldsymbol{j}=\mathbf{1}, \ldots, \boldsymbol{b}$

## 无界背包问题

每种物体能够无限选择;

复杂度分析: O(nb)

首先定义动态规划的 dp 数组含义, 将 $dp_{\{i, j\}}$定义为:在 1-i 的索引范围中挑选物体, 在最大负重为 j 时, 背包内物体的最大价值;

1) 当 i == 0 时, 无法挑选任何物体, 所以初始化为 0;

2) i > 0 时, 可知状态转移方程:

$d p^{\{i, j\}}=\max \left(d p^{\{i-1, j\}}, d p^{\{i, j-w[i]\}}+v[i]\right)$
$i=1, \ldots, n ; j=1, \ldots, b$


## 投资问题整理

每个项目不能重复投资，产生的效益为非负数 m 元钱全用来投资,即

$\sum_{i=1}^{n} x_{i}=m$

### 解题思路

动归求解

状态 dp[item][tot_m] 表示前 item 个项目，共投资 tot_m，能得到的最大收益。

状态转移方程:

dp[item][tot_m] = max(dp[item - 1][tot_m - now_m]), tot_m $\epsilon$ e [0,m], now_m $\epsilon$ [0,tot_m]

初始化:

dp[O][tot_m] = fun[O][tot_m], tot_m $\epsilon$ [0,m]

```
def invest(money, num_items, fun):
    dp = [[0 for col in range(money+1)] for row in range(num_items)]
    for tot_m in range(money + 1):
        dp[0][tot_m] = fun[0][tot_m]
    for item in range(1, num_items):
        for tot_m in range(money + 1):
            for now_m in range(tot_m + 1):
                dp [item] [tot_m]=max(dp[item][tot_m], dp[item - 1][tot_m - now_m] + fun[item] [now_m])
    return dp[num_items-1] [money]

if __name__ == '__main__':
    money = 5
    num_items = 4
    fun = [[0, 20, 1, 9, 12, 5],
           [0, 0, 3, 10, 4, 32],
           [0, 14, 6, 20, 17, 33],
           [0, 15, 16, 18, 20, 22]]
    print('MaxIncome = ', invest(money, num_items, fun))
```


## 编辑距离, LC 72, LC 312

### LC 72. Edit Distance 编辑距离

Given two strings word1 and word2, return the minimum number of operations required to convert word1 to word2.

You have the following three operations permitted on a word:

- Insert a character

- Delete a character

- Replace a character

给你两个单词 word1 和 word2， 请返回将 word1 转换成 word2 所使用的最少操作数 。

你可以对一个单词进行如下三种操作：

- 插入一个字符

- 删除一个字符

- 替换一个字符

其编辑距离算法概念是俄罗斯科学家弗拉基米尔·莱文斯坦在1965年提出。

- DNA 分析

- 拼写检查 

- 抄袭识别(比如论文查重) 

- 语音识别

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
```

![](w7-kp-1.png)

![](w7-kp-2.png)

![](w7-kp-3.png)


```
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
```

![](w7-kp-4.png)

```
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

### LC 312. Burst Balloons 戳气球


You are given n balloons, indexed from 0 to n - 1. Each balloon is painted with a number on it represented by an array nums. You are asked to burst all the balloons.

If you burst the ith balloon, you will get nums[i - 1] * nums[i] * nums[i + 1] coins. If i - 1 or i + 1 goes out of bounds of the array, then treat it as if there is a balloon with a 1 painted on it.

Return the maximum coins you can collect by bursting the balloons wisely.

有 n 个气球，编号为0 到 n - 1，每个气球上都标有一个数字，这些数字存在数组 nums 中。

现在要求你戳破所有的气球。戳破第 i 个气球，你可以获得 nums[i - 1] * nums[i] * nums[i + 1] 枚硬币。这里的 i - 1 和 i + 1 代表和 i 相邻的两个气球的序号。如果 i - 1或 i + 1 超出了数组的边界，那么就当它是一个数字为 1 的气球。

求所能获得硬币的最大数量。

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
        # print(dp)
        return dp[0][-1]
```

![](w7-kp-5.png)

![](w7-kp-6.png)

![](w7-kp-7.png)

```
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
```

```
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

## 樹

- 要理解树数据结构的含义以及如何使用。

- 使用列表实现树。

- 使用类和引用来实现树。

- 实现树作为递归数据结构。

- 使用堆实现优先级队列。

### 實例

- 文件系统树

- 网页

- 句子

- 公式解析

![](w8-kp-1.png)

### 樹各名詞定義

![](w8-kp-2.png)

### 定义树（Tree）

定义一：树由一组节点和一组连接节点的边组成。且树具有以下属性：

- 树的一个节点被指定为根节点
- 除了根节点之外，每个节点 n 通过一个其他节点 p 的边连接，其中 p 是 n 的父节点
- 从根路径遍历到每个节点路径唯一
- 如果树中的每个节点最多有两个子节点，我们说该树是一个二叉树

定义二：树是空的，或者由一个根节点和零个或多个子树组成，每个子树也是一棵树。每个子树的根节点通过边连接到父树的根节点。

![](w8-kp-3.png)

### Know Thy Complexities!

Reference: https://www.bigocheatsheet.com/

![](w8-kp-4.png)

![](w8-kp-5.png)

### Binary Tree & Binary Search Tree

- Binary Tree : 基于二叉堆实现优先队列
    
- Binary Search Tree : 二叉查找树

![](w8-kp-6.png)

### 树的表示-列表

![](w8-kp-7.png)

```
myTree = ['a', #root
    ['b', #left subtree
        ['d', [], []],
        ['e', [], []] ],
    ['c', #right subtree
        ['f', [], []],
        [] ]
    ]

print(myTree)
print('left subtree = ', myTree[1])
print('root = ', myTree[0])
print('right subtree = ', myTree[2])
```
```
['a', ['b', ['d', [], []], ['e', [], []]], ['c', ['f', [], []], []]]
left subtree =  ['b', ['d', [], []], ['e', [], []]]
root =  a
right subtree =  ['c', ['f', [], []], []]
```

```
def BinaryTree(r):
    return [r, [], []]

def insertLeft(root,newBranch):
    t = root.pop(1)
    if len(t) > 1:
        root.insert(1,[newBranch,t,[]])
    else:
        root.insert(1,[newBranch, [], []])
    return root

def insertRight(root,newBranch):
    t = root.pop(2)
    if len(t) > 1:
        root.insert(2,[newBranch,[],t])
    else:
        root.insert(2,[newBranch,[],[]])
    return root

def getRootVal(root):
    return root[0]

def setRootVal(root,newVal):
    root[0] = newVal

def getLeftChild(root):
    return root[1]

def getRightChild(root):
    return root[2]

r = BinaryTree(3)
insertLeft(r,4)
insertLeft(r,5)
insertRight(r,6)
insertRight(r,7)
l = getLeftChild(r)
print(l)
setRootVal(l,9)
print(r)
insertLeft(l,11)
print(r)
print(getRightChild(getRightChild(r)))
```

```
[5, [4, [], []], []]
[3, [9, [4, [], []], []], [7, [], [6, [], []]]]
[3, [9, [11, [4, [], []], []], []], [7, [], [6, [], []]]]
[6, [], []]
```

### 树的表示-类

```
class BinaryTree:
    def __init__(self,rootObj):
        self.key = rootObj
        self.leftChild = None
        self.rightChild = None
    def insertLeft(self,newNode):
        if self.leftChild == None:
            self.leftChild = BinaryTree(newNode)
        else:
            t = BinaryTree(newNode)
            t.leftChild = self.leftChild
            self.leftChild = t
    def insertRight(self,newNode):
        if self.rightChild == None:
            self.rightChild = BinaryTree(newNode)
        else:
            t = BinaryTree(newNode)
            t.rightChild = self.rightChild
            self.rightChild = t
    def getRightChild(self):
        return self.rightChild
    def getLeftChild(self):
        return self.leftChild
    def setRootVal(self,obj):
        self.key = obj
    def getRootVal(self):
        return self.key

r = BinaryTree('a')
print(r.getRootVal())
print(r.getLeftChild())
r.insertLeft('b')
print(r.getLeftChild())
print(r.getLeftChild().getRootVal())
r.insertRight('c')
print(r.getRightChild())
print(r.getRightChild().getRootVal())
r.getRightChild().setRootVal('hello')
print(r.getRightChild().getRootVal())
```
```
a
None
<__main__.BinaryTree object at 0x000002A248647A30>
b
<__main__.BinaryTree object at 0x000002A248591580>
c
hello
```

### 树和链表, LC 100, LC 112, LC 226

```
class Node:
    def __init__(self, initdata):
        self.data = initdata
        self.next = None
    def getData(self):
        return self.data
    def getNext(self):
        return self.next
    def setData(self,newdata):
        self.data = newdata
    def setNext(self,newnext):
        self.next = newnext

class BinaryTree:
    def __init__(self,rootObj):
        self.key = rootObj
        self.leftChild = None
        self.rightChild = None
    def insertLeft(self,newNode):
        if self.leftChild == None:
            self.leftChild = BinaryTree(newNode)
        else:
            t = BinaryTree(newNode)
            t.leftChild = self.leftChild
            self.leftChild = t
    def insertRight(self,newNode):
        if self.rightChild == None:
            self.rightChild = BinaryTree(newNode)
        else:
            t = BinaryTree(newNode)
            t.rightChild = self.rightChild
            self.rightChild = t
```

1. LC 100 Same Tree 相同的树

Given the roots of two binary trees p and q, write a function to check if they are the same or not.

Two binary trees are considered the same if they are structurally identical, and the nodes have the same value.


给你两棵二叉树的根节点 p 和 q ，编写一个函数来检验这两棵树是否相同。

如果两个树在结构上相同，并且节点具有相同的值，则认为它们是相同的。

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

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None
class Solution(object):
    def isSameTree(self, p, q):
        if p is None and q is None:
            return True
        if p is not None and q is not None:
            return p.val == q.val and self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
        return False
if __name__ == "__main__":
    root1, root1.left, root1.right = TreeNode(1), TreeNode(2), TreeNode(3)
    root2, root2.left, root2.right = TreeNode(1), TreeNode(2), TreeNode(3)
    print(Solution().isSameTree(root1, root2))
```

2. LC 112 Path Sum 路径总和

Given the root of a binary tree and an integer targetSum, return true if the tree has a root-to-leaf path such that adding up all the values along the path equals targetSum.

A leaf is a node with no children.

给你二叉树的根节点 root 和一个表示目标和的整数 targetSum 。判断该树中是否存在 根节点到叶子节点 的路径，这条路径上所有节点值相加等于目标和 targetSum 。如果存在，返回 true ；否则，返回 false 。

叶子节点 是指没有子节点的节点。
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
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution(object):
    def hasPathSum(self, root, sum):
        if not root:
            return False
        if not root.left and not root.right and root.val == sum:
            return True
        sum -= root.val
        return self.hasPathSum(root.left, sum) or self.hasPathSum(root.right, sum)

if __name__ == '__main__':
    root = TreeNode(5)
    root.left = TreeNode(4)
    root.right = TreeNode(8)
    root.right.left = TreeNode(13)
    root.right.right = TreeNode(4)
    root.right.right.right = TreeNode(1)
    root.left.left = TreeNode(11)
    root.left.left.left = TreeNode(7)
    root.left.left.right = TreeNode(2)
    print(Solution().hasPathSum(root, 22))
```
3. LC 226

Given the root of a binary tree, invert the tree, and return its root.

给你一棵二叉树的根节点 root ，翻转这棵二叉树，并返回其根节点。

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

### 树的遍历(Traversal), LC 144

- 1. 前序(Pre-order)：根-左-右

- 2. 中序(In-order)：左-根-右

- 3. 后序(Post-order)：左-右-根

![](w8-kp-8.png)

![](w8-kp-9.png)

```
# 树的遍历

def preorder(tree):
    if tree:
        print(tree.getRootVal())
        preorder(tree.getLeftChild())
        preorder(tree.getRightChild())

def postorder(tree):
    if tree != None:
        postorder(tree.getLeftChild())
        postorder(tree.getRightChild())
        print(tree.getRootVal())

def inorder(tree):
    if tree != None:
        inorder(tree.getLeftChild())
        print(tree.getRootVal())
        inorder(tree.getRightChild())
```
1. LC 144 Binary Tree Preorder Traversal 二叉树的前序遍历

Given the root of a binary tree, return the preorder traversal of its nodes' values.

给你二叉树的根节点 root ，返回它节点值的 前序 遍历。

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
### 分析树 (Parse Tree)

![](w8-kp-11.png)

1. 构建分析树： （3 +（4 * 5））

四种不同的符号要考虑：左括号，右括号，运算符和操作数


2. 定义四个规则如下：

- (1) 如果当前符号是 '('，添加一个新节点作为当前节点的左子节点，并下降到左子节点。

- (2) 如果当前符号在列表 `['+'，' - '，'/'，'*']` 中，请将当前节点的根值设置为由当前符号表示的运算符。 添加一个新节点作为当前节点的右子节点，并下降到右子节点。

- (3) 如果当前符号是数字，请将当前节点的根值设置为该数字并返回到父节点。

- (4) 如果当前符号是 ')'，则转到当前节点的父节点。

### 构建分析树

```
['('，'3'，'+'，'('，'4'，'*'，'5'，')'，')']
```

1. 创建一个空树。

2. 读取 ( 作为第一个标记。按规则1，创建一个新节点作为根的左子节点。使当前节点到这个新子节点。

3. 读取 3 作为下一个符号。按照规则3，将当前节点的根值设置为3，使当前节点返回到父节点。

4. 读取 + 作为下一个符号。根据规则2，将当前节点的根值设置为+，并添加一个新节点作为右子节点。新的右子节点成为当前节点。

5. 读取 ( 作为下一个符号，按规则1，创建一个新节点作为当前节点的左子节点，新的左子节点成为当前节点。

6. 读取 4 作为下一个符号。根据规则3，将当前节点的值设置为 4。使当前节点返回到父节点。

7. 读取 * 作为下一个符号。根据规则2，将当前节点的根值设置为 `*`，并创建一个新的右子节点。新的右子节点成为当前节点。

8. 读取 5 作为下一个符号。根据规则3，将当前节点的根值设置为5。使当前节点返回到父节点。

9. 读取 ) 作为下一个符号。根据规则4，当前节点返回到父节点。

10. 读取 ) 作为下一个符号。根据规则4，当前节点返回到父节点 + 。没有+的父节点，完成创建。

![](w8-kp-12.png)

```
def buildParseTree(fpexp):
    fplist = fpexp.split()
    pStack = Stack()
    eTree = BinaryTree('')
    pStack.push(eTree)
    currentTree = eTree
    for i in fplist:
        if i == '(':
            currentTree.insertLeft('')
            pStack.push(currentTree)
            currentTree = currentTree.getLeftChild()
        elif i not in ['+', '-', '*', '/', ')']:
            currentTree.setRootVal(int(i))
            parent = pStack.pop()
            currentTree = parent
        elif i in ['+', '-', '*', '/']:
            currentTree.setRootVal(i)
            currentTree.insertRight('')
            pStack.push(currentTree)
            currentTree = currentTree.getRightChild()
        elif i == ')':
            currentTree = pStack.pop()
        else:
            raise ValueError
    return eTree
```

![](w8-kp-13.png)

```
import operator
def evaluate(parseTree):
    opers = {'+':operator.add, '-':operator.sub, '*':operator.mul, '/':operator.truediv}
    leftC = parseTree.getLeftChild()
    rightC = parseTree.getRightChild()
    if leftC and rightC:
        fn = opers[parseTree.getRootVal()]
        return fn(evaluate(leftC),evaluate(rightC))
    else:
        return parseTree.getRootVal()
```

### 树的遍历 & 分析树 (Parse Tree) LC 105

1. LC 105. Construct Binary Tree from Preorder and Inorder Traversal 从前序与中序遍历序列构造二叉树

iven two integer arrays preorder and inorder where preorder is the preorder traversal of a binary tree and inorder is the inorder traversal of the same tree, construct and return the binary tree.

给定两个整数数组 preorder 和 inorder ，其中 preorder 是二叉树的先序遍历， inorder 是同一棵树的中序遍历，请构造二叉树并返回其根节点。

Example 1:

```
Input: preorder = [3,9,20,15,7], inorder = [9,3,15,20,7]
Output: [3,9,20,null,null,15,7]
```

Example 2:

```
Input: preorder = [-1], inorder = [-1]
Output: [-1]
```

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
```

Reference

https://www.youtube.com/watch?v=GeltTz3Z1rw

LC 105 說明

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

## 基于二叉堆实现优先队列

### 优先队列 priority queue

1. 优先队列

- 普通的队列是一种先进先出的数据结构，元素在队列尾追加，而从队列头删除。

- 在优先队列中，元素被赋予优先级。当访问元素时，具有最高优先级的元素最先删除。

2. 优先队列不再遵循先入先出的原则，而是分为两种情况：

- 最大优先队列，无论入队顺序，当前最大的元素优先出队。

- 最小优先队列，无论入队顺序，当前最小的元素优先出队。

3. 最小优先队列例子

![](w9-kp-10.png)

### 二叉堆

1. 二叉堆本质上是一种完全二叉树，它分为两个类型：

- 最大堆

- 最小堆

![](w9-kp-11.png)

2. 什么是二叉堆？

- 最大堆任何一个父节点的值，都大于等于它左右孩子节点的值。

- 最小堆任何一个父节点的值，都小于等于它左右孩子节点的值。

### 二叉堆操作

最小二叉堆实现的基本操作如下：

- BinHeap() 创建一个新的，空的二叉堆。

- insert(k) 向堆添加一个新项。

- findMin() 返回具有最小键值的项，并将项留在堆中。

- delMin() 返回具有最小键值的项，从堆中删除该项。

- 如果堆是空的，isEmpty() 返回 true，否则返回 false。

- size() 返回堆中的项数。

- buildHeap(list) 从键列表构建一个新的堆。

```
bh = BinHeap()
print(bh.heapList)
bh.buildHeap([9, 5, 6, 2, 3])
print(bh.heapList)
print(bh.delMin())
print(bh.delMin())
print(bh.delMin())
print(bh.delMin())
print(bh.delMin())
```

### 二叉堆实现

1. 结构属性

完全二叉树的一个属性是：用列表表示，则父级和子级之间是 2p 和 2p+1 关系。

2. 排序属性

堆的排序属性如下： 在堆中，对于具有父 p 的每个节点 x， p 中的键小于或等于 x 中的键。

![](w9-kp-12.png)

3. 堆操作

(1). 构造函数：

整个二叉堆由单个列表表示， 所以构造函数将初始化列表和一个 currentSize 属性来跟踪堆的当前大小。

```
class BinHeap:
    def __init__(self):
        self.heapList = [0]
        self.currentSize = 0
```

(2). insert(k)：

- 将项添加到列表中最简单最有效的方法是将项附加到列表的末尾 ，但必须维护满足结构属性和排序属性。

- 方案：通过比较新添加的项与其父项，如果新添加的项小于其父项， 则将项与其父项交换。

![](w9-kp-13.png)

```
def insert(self, k):
    self.heapList.append(k)
    self.currentSize = self.currentSize + 1
    self.percUp(self.currentSize)
```

```
def percUp(self, i):
    while i // 2 > 0:
        if self.heapList[i] < self.heapList[i // 2]:
            tmp = self.heapList[i // 2]
            self.heapList[i // 2] = self.heapList[i]
            self.heapList[i] = tmp
        i = i // 2
```
(3). delMin()：

两步保持结构属性和顺序属性：

- 获取列表中的最后一个项并将其移动到根位置保持堆结构属性

- 从新根节点开始，依次向下和最小的子节点交换保持顺序属性

![](w9-kp-14.png)

```
def delMin(self):
    retval = self.heapList[1]
    self.heapList[1] = self.heapList[self.currentSize]
    self.currentSize = self.currentSize - 1
    self.heapList.pop()
    self.percDown(1)
    return retval

def percDown(self, i):
    while (i * 2) <= self.currentSize:
        mc = self.minChild(i)
        if self.heapList[i] > self.heapList[mc]:
            tmp = self.heapList[i]
            self.heapList[i] = self.heapList[mc]
            self.heapList[mc] = tmp
        i = mc

def minChild(self, i):
    if i * 2 + 1 > self.currentSize:
        return i * 2
    else:
        if self.heapList[i*2] < self.heapList[i*2+1]:
            return i * 2
        else:
            return i * 2 + 1
```

### 构建二叉堆操作

1. buildHeap(alist)：

```
def buildHeap(self, alist):
    i = len(alist) // 2
    self.currentSize = len(alist)
    self.heapList = [0] + alist[:]
    while (i > 0):
        self.percDown(i)
        i = i - 1
```

![](w9-kp-15.png)

2. 复杂度

```
O(nlogn) or O(n) ?
```

3. Code

```
def buildHeap(self, alist):
    i = len(alist) // 2
    self.currentSize = len(alist)
    self.heapList = [0] + alist[:]
    while (i > 0):
        self.percDown(i)
        i = i - 1
```

4. Tip

设计一个找到数据流中第K大元素的类（class）。注意是排序后的第 K 大元素，不是第 K 个不同的元素。你的 KthLargest 类需要一个同时接收整数 k 和整数数组 nums 的构造器，它包含数据流中的初始元素。

每次调用 KthLargest.add，返回当前数据流中第 K 大的元素。

```
int k = 3;
int[] arr = [4,5,8,2];
KthLargest kthLargest = new KthLargest(3, arr);
kthLargest.add(3); // returns 4
kthLargest.add(5); // returns 5
kthLargest.add(10); // returns 5
kthLargest.add(9); // returns 8
kthLargest.add(4); // returns 8
```

说明: 你可以假设 nums 的长度 ≥ k-1 且 k ≥ 1。

```
import heapq
h = []
heapq.heappush(h,5)
heapq.heappush(h,2)
heapq.heappush(h,8)
heapq.heappush(h,4)
print(heapq.heappop(h))
print(heapq.heappop(h))
print(heapq.heappop(h))
print(heapq.heappop(h))
h = [9,8,7,6,2,4,5]
print(h)
heapq.heapify(h)
print(h)
```

```
2
4
5
8
[9, 8, 7, 6, 2, 4, 5]
[2, 6, 4, 9, 8, 7, 5]
```

```
class KthLargest(object):
    def __init__(self, k, nums):
        self.pool = nums
        self.size = len(self.pool)
        self.k = k
        heapq.heapify(self.pool)
        while self.size > k:
            heapq.heappop(self.pool)
            self.size -= 1
    def add(self, val):
        if self.size < self.k:
            heapq.heappush(self.pool, val)
            self.size += 1
        elif val > self.pool[0]:
            heapq.heapreplace(self.pool, val)
        return self.pool[0]

k = 3
arr = [4,5,8,2]
kthLargest = KthLargest(3, arr)
print(kthLargest.add(3))
print(kthLargest.add(5))
print(kthLargest.add(10))
print(kthLargest.add(9))
print(kthLargest.add(4))
```

```
4
5
5
8
8
```

### 二叉查找树

1. 二叉查找树（又叫作二叉搜索树或二叉排序树）是一种数据结构 。其二叉查找树的兩個性質，且每个结点最多有两个子结点。

![](w9-kp-16.png)


2. 第一个是每个结点的值均大于其左子树上任意一个结点的值。比如结点 9 大于其左子树上的3和 8。

3. 第二个是每个结点的值均小于其右子树上任意一个结点的值。比如结点 15 小于其右子树上的 23、17 和 28。

4. 根据这两个性质可以得到以下结论。首先，二叉查找树的最小结点要从顶端开始，往其左下的末端寻找。此处最小值为 3。

5. 反过来，二叉查找树的最大结点要从顶端开始，往其右下的末端寻找。此处最大值为 28。

- Map() 创建一个新的空 map 。

- put(key，val) 向 map 中添加一个新的键值对。 如果键已经在 map 中，那么用新值替换旧值。

- get(key) 给定一个键， 返回存储在 map 中的值， 否则为 None。

- del 使用 del map[key] 形式的语句从 map 中删除键值对。

- len() 返回存储在映射中的键值对的数量。

- in 返回 True 如果给定的键在 map 中。

6. 实现

```
class BinarySearchTree:
    def __init__(self):
        self.root = None
        self.size = 0
    def length(self):
        return self.size
    def __len__(self):
        return self.size

class TreeNode:
    def __init__(self,key,val,left=None,right=None,parent=None):
        self.key = key
        self.payload = val
        self.leftChild = left
        self.rightChild = right
        self.parent = parent
    def hasLeftChild(self):
        return self.leftChild
    def hasRightChild(self):
        return self.rightChild
    def isLeftChild(self):
        return self.parent and self.parent.leftChild == self
    def isRightChild(self):
        return self.parent and self.parent.rightChild == self

    def put(self,key,val):
        if self.root:
            self._put(key,val,self.root)
        else:
            self.root = TreeNode(key,val)
        self.size = self.size + 1
    def _put(self,key,val,currentNode):
        if key < currentNode.key:
            if currentNode.hasLeftChild():
                self._put(key,val,currentNode.leftChild)
            else:
                currentNode.leftChild = TreeNode(key,val,parent=currentNode)
        else:
            if currentNode.hasRightChild():
                self._put(key,val,currentNode.rightChild)
            else:
                currentNode.rightChild = TreeNode(key,val,parent=currentNode)
```

7. 二叉查找树 LC 235

Given a binary search tree (BST), find the lowest common ancestor (LCA) of two given nodes in the BST.

According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined between two nodes p and q as the lowest node in T that has both p and q as descendants (where we allow a node to be a descendant of itself).”

给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。

百度百科中最近公共祖先的定义为：“对于有根树 T 的两个结点 p、q，最近公共祖先表示为一个结点 x，满足 x 是 p、q 的祖先且 x 的深度尽可能大（一个节点也可以是它自己的祖先）。”

例如，给定如下二叉搜索树: root = [6,2,8,0,4,7,9,null,null,3,5]

https://baike.baidu.com/item/%E6%9C%80%E8%BF%91%E5%85%AC%E5%85%B1%E7%A5%96%E5%85%88/8918834

https://en.wikipedia.org/wiki/Lowest_common_ancestor

Example 1:

```
Input: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8
Output: 6
Explanation: The LCA of nodes 2 and 8 is 6.
节点 2 和节点 8 的最近公共祖先是 6。
```

Example 2:

```
Input: root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 4
Output: 2
Explanation: The LCA of nodes 2 and 4 is 2, since a node can be a descendant of itself according to the LCA definition.
节点 2 和节点 4 的最近公共祖先是 2, 因为根据定义最近公共祖先节点可以为节点本身。
```

Example 3:

```
Input: root = [2,1], p = 2, q = 1
Output: 2
```

Constraints:

- The number of nodes in the tree is in the range $[2, 10^{5}]$.

- $-109$ <= Node.val <= $10^{9}$

- All Node.val are unique.

- p != q

- p and q will exist in the BST.

- 所有节点的值都是唯一的。

- p、q 为不同节点且均存在于给定的二叉搜索树中。

解题思路

1. 在二叉搜索树中求两个节点的最近公共祖先，由于二叉搜索树的特殊性质，所以找任意两个节点的最近公共祖先非常简单。

2. python3 利用二叉搜索树的特点，如果p、q的值都小于root，说明p q 肯定在root的左子树中；如果p q都大于root，说明肯定在root的右子树中，如果一个在左一个在右 则说明此时的root记为对应的最近公共祖先

```
class Solution(object):
    def lowestCommonAncestor(self, root, p, q):
        """
        :type root: TreeNode
        :type p: TreeNode
        :type q: TreeNode
        :rtype: TreeNode
        """
        if p.val<root.val and q.val<root.val:
            return self.lowestCommonAncestor(root.left,p,q)
        if p.val>root.val and q.val>root.val:
            return self.lowestCommonAncestor(root.right,p,q)
```

## Trie 字典树

![](w10-kp-2.png)

- Trie 树的基本结构

- Trie 树的核心思想

- Trie 树的基本性质

- Trie 树的实现

1. 实际问题

![](w10-kp-1.png)

2. 基本结构

Trie 树,即字典树,又称单词查找树或键树,是一种树形结构,是一种哈希树的变种。典型应用是用于统计和排序大量的字符串但不仅限于字符串,所以经常被搜索引擎系统用于文本词频统计。

它的优点是:最大限度地减少无谓的字符串比较,查询效率比哈希表高。

3. 核心思想

Trie 的核心思想空间换时间。利用字符串的公共前缀来降低查询时间的开销以达到提高效率的目的。

4. 基本性质

1.根节点不包含字符,除根节点外每一个节点都只包含一个字符。

2.从根节点到某一节点,路径上经过的字符连接起来,为该节点对应的字符串。

3.每个节点的所有子节点包含的字符都不相同。

5. 实现

LC 208. Implement Trie (Prefix Tree) 实现 Trie (前缀树)

A trie (pronounced as "try") or prefix tree is a tree data structure used to efficiently store and retrieve keys in a dataset of strings. There are various applications of this data structure, such as autocomplete and spellchecker.

Implement the Trie class:

- Trie() Initializes the trie object.

- void insert(String word) Inserts the string word into the trie.

- boolean search(String word) Returns true if the string word is in the trie (i.e., was inserted before), and false otherwise.

- boolean startsWith(String prefix) Returns true if there is a previously inserted string word that has the prefix prefix, and false otherwise.

Trie（发音类似 "try"）或者说 前缀树 是一种树形数据结构，用于高效地存储和检索字符串数据集中的键。这一数据结构有相当多的应用情景，例如自动补完和拼写检查。

请你实现 Trie 类：

Trie() 初始化前缀树对象。
void insert(String word) 向前缀树中插入字符串 word 。
boolean search(String word) 如果字符串 word 在前缀树中，返回 true（即，在检索之前已经插入）；否则，返回 false 。
boolean startsWith(String prefix) 如果之前已经插入的字符串 word 的前缀之一为 prefix ，返回 true ；否则，返回 false 。

Example 1:

```
Input
["Trie", "insert", "search", "search", "startsWith", "insert", "search"]
[[], ["apple"], ["apple"], ["app"], ["app"], ["app"], ["app"]]
Output
[null, null, true, false, true, null, true]

Explanation
Trie trie = new Trie();
trie.insert("apple");
trie.search("apple");   // return True
trie.search("app");     // return False
trie.startsWith("app"); // return True
trie.insert("app");
trie.search("app");     // return True
```

Constraints:

- 1 <= word.length, prefix.length <= 2000

- word and prefix consist only of lowercase English letters.

word 和 prefix 仅由小写英文字母组成

- At most 3 * $10^{4}$ calls in total will be made to insert, search, and startsWith.

insert、search 和 startsWith 调用次数 总计 不超过 3 * $10^4$ 次

```
class TreeNode(object):
    def __init__(self):
        self.word = False
        self.children = {}

class Trie(object):
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = TreeNode()
    def insert(self, word):
        """
        Inserts a word into the trie.
        :type word: str
        :rtype: void
        """
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TreeNode()
            node = node.children[char]
        node.word = True

    def search(self, word):
        """
        Returns if the word is in the trie.
        :type word: str
        :rtype: bool
        """
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.word
    def startsWith(self, prefix):
        """
        Returns if there is any word in the trie
        that starts with the given prefix.
        :type prefix: str
        :rtype: bool
        """
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True
```

LC 720. Longest Word in Dictionary 词典中最长的单词

Given an array of strings words representing an English Dictionary, return the longest word in words that can be built one character at a time by other words in words.

If there is more than one possible answer, return the longest word with the smallest lexicographical order. If there is no answer, return the empty string.


给出一个字符串数组 words 组成的一本英语词典。返回 words 中最长的一个单词，该单词是由 words 词典中其他单词逐步添加一个字母组成。

若其中有多个可行的答案，则返回答案中字典序最小的单词。若无答案，则返回空字符串。



Example 1:

```
Input: words = ["w","wo","wor","worl","world"]
Output: "world"
Explanation: The word "world" can be built one character at a time by "w", "wo", "wor", and "worl".
单词"world"可由"w", "wo", "wor", 和 "worl"逐步添加一个字母组成。
```

Example 2:

```
Input: words = ["a","banana","app","appl","ap","apply","apple"]
Output: "apple"
Explanation: Both "apply" and "apple" can be built from other words in the dictionary. However, "apple" is lexicographically smaller than "apply".
"apply" 和 "apple" 都能由词典中的单词组成。但是 "apple" 的字典序小于 "apply" 
```


Constraints:

- 1 <= words.length <= 1000

- 1 <= words[i].length <= 30

- words[i] consists of lowercase English letters.

所有输入的字符串 words[i] 都只包含小写字母。

```
class Solution(object):
    def longestWord(self, words):
        valid = set([""])
        for word in sorted(words, key=len):
            if word[:-1] in valid:
                valid.add(word)
        return max(sorted(valid), key=len)
if __name__ == '__main__':
    words = ["a", "banana", "app", "appl", "ap", "apply", "apple"]
    print(Solution().longestWord(words))
```

## LC 347 (桶排序) Top K Frequent Elements 前 K 个高频元素

Given an integer array nums and an integer k, return the k most frequent elements. You may return the answer in any order.

给你一个整数数组 nums 和一个整数 k ，请你返回其中出现频率前 k 高的元素。你可以按 任意顺序 返回答案。

Example 1:

```
Input: nums = [1,1,1,2,2,3], k = 2
Output: [1,2]
```

Example 2:

```
Input: nums = [1], k = 1
Output: [1]
```


Constraints:

- $1 <= nums.length <= 10^{5}$

- k is in the range [1, the number of unique elements in the array].

- It is guaranteed that the answer is unique.

Reference :

https://zh.m.wikipedia.org/zh-hant/%E6%A1%B6%E6%8E%92%E5%BA%8F

0.

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

1. 简单排序

时间复杂度O(nlogn)

```
from typing import List
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        counts = list(collections.Counter(nums).items())
        counts.sort(key=lambda X: x[1], reverse=True)
        return [count[0] for count in counts[:k]]
```

2. 小根堆

```
from typing import List
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        counts = list(collections.Counter(nums).items())
        heap = []
        def sift_down(heap, root, k):
            tmp = heap[root]
            while root << 1 < k:
                child = root << 1
                if child + 1 < k and heap[child + 1][1] < heap[child][1]:
                    child += 1
                if heap[child][1] < tmp[1]:
                    heap[root] = heap[child]
                    root = child
                else:
                    break
            heap[root] = tmp
        def sift_up(heap, child):
            tmp = heap[child]
            while child >> 1 > 0 and tmp[1] < heap[child >> 1][1]:
                heap[child] = heap[child >> 1]
                child >>= 1
            heap[child] = tmp
        heap = [(0, 0)]
        for i in range(k):
            heap.append(counts[i])
            sift_up(heap, len(heap) - 1)
        for i in range(k, len(counts)):
            if counts[i][1] > heap[1][1]:
                heap[1] = counts[i]
                sift_down(heap, 1, k + 1)
        return [item[0] for item in heap[1:]]
```

3. 桶排序

使用桶排序的方式,倒序遍历桶,得到 k 个值后结束,时间复杂度为 O(n)

```
from typing import List
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        counts = collections.Counter (nums)
        bucket = dict()
        for key, value in counts.items():
            if value not in bucket:
                bucket[value] = [key]
            else:
                bucket[value].append(key)
        res = []
        for value in range(len(nums), -1, -1):
            if len(res) >= k:
                break
            if value in bucket:
                res.extend(bucket[value])
        return res
```

## 圖 & LC 997 (KP)

![](w12-kp-1.png)

![](w12-kp-2.png)

- Graph() 创建一个新的空图。

- addVertex(vert) 向图中添加一个顶点实例。

- addEdge(fromVert, toVert) 向连接两个顶点的图添加一个新的有向边。

- addEdge(fromVert, toVert, weight) 向连接两个顶点的图添加一个新的加权的有向边。

- getVertex(vertKey) 在图中找到名为 vertKey 的顶点。

- getVertices() 返回图中所有顶点的列表。

- in 返回 True 如果 vertex in graph 里给定的顶点在图中，否则返回False。

![](w12-kp-3.png)

```
class Vertex:
    def __init__(self, key):
        self.id = key
        self.connectedTo = {}

    def addNeighbor(self,nbr,weight=0):
        self.connectedTo[nbr] = weight

    def __str__(self):
        return str(self.id) + ' connectedTo: ' + str([x.id for x in self.connectedTo])

    def getConnections(self):
        return self.connectedTo.keys()

    def getId(self):
        return self.id

    def getWeight(self,nbr):
        return self.connectedTo[nbr]
class Graph:
    def __init__(self):
        self.vertList = {}
        self.numVertices = 0

    def addVertex(self, key):
        self.numVertices = self.numVertices + 1
        newVertex = Vertex(key)
        self.vertList[key] = newVertex
        return newVertex

    def getVertex(self,n):
        if n in self.vertList:
            return self.vertList[n]
        else:
            return None

    def __contains__(self,n):
        return n in self.vertList

    def addEdge(self, f, t, cost=0):
        if f not in self.vertList:
            nv = self.addVertex(f)
        if t not in self.vertList:
            nv = self.addVertex(t)
        self.vertList[f].addNeighbor(self.vertList[t], cost)

    def getVertices(self):
        return self.vertList.keys()

    def __iter__(self):
        return iter(self.vertList.values())
g = Graph()
for i in range(6):
    g.addVertex(i)
print(g.vertList)
g.addEdge(0, 1, 5)
g.addEdge(0, 5, 2)
g.addEdge(1, 2, 4)
g.addEdge(2, 3, 9)
g.addEdge(3, 4, 7)
g.addEdge(3, 5, 3)
g.addEdge(4, 0, 1)
g.addEdge(5, 4, 8)
g.addEdge(5, 2, 1)
g.addEdge(6, 2, 1)

for v in g:
    for w in v.getConnections():
        print("( %s , %s )" % (v.getId(), w.getId()))
```

### LC 997. Find the Town Judge 找到小镇的法官

In a town, there are n people labeled from 1 to n. There is a rumor that one of these people is secretly the town judge.

If the town judge exists, then:

1. The town judge trusts nobody.

2. Everybody (except for the town judge) trusts the town judge.

3. There is exactly one person that satisfies properties 1 and 2.

You are given an array trust where trust[i] = [$a_i$, $b_i$] representing that the person labeled ai trusts the person labeled bi.

Return the label of the town judge if the town judge exists and can be identified, or return -1 otherwise.

小镇里有 n 个人，按从 1 到 n 的顺序编号。传言称，这些人中有一个暗地里是小镇法官。

如果小镇法官真的存在，那么：

小镇法官不会信任任何人。
每个人（除了小镇法官）都信任这位小镇法官。
只有一个人同时满足属性 1 和属性 2 。
给你一个数组 trust ，其中 trust[i] = [$a_i$, $b_i$] 表示编号为 $a_i$ 的人信任编号为 $b_i$ 的人。

如果小镇法官存在并且可以确定他的身份，请返回该法官的编号；否则，返回 -1 。


Example 1:

```
Input: n = 2, trust = [[1,2]]
Output: 2
```

Example 2:

```
Input: n = 3, trust = [[1,3],[2,3]]
Output: 3
```

Example 3:

```
Input: n = 3, trust = [[1,3],[2,3],[3,1]]
Output: -1
```

Constraints:

- 1 <= n <= 1000

- 0 <= trust.length <= $10^4$

- trust[i].length == 2

- All the pairs of trust are unique. (trust 中的所有 trust[i] = [$a_i$, $b_i$] 互不相同)

- $a_i$ != $b_i$

- 1 <= $a_i$, $b_i$ <= n

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


if __name__ == '__main__':
    trust = [[1, 3], [2, 3]]
    # trust = [[1,3],[2,1], [2,1]]
    # trust = [[1,8],[1,3],[2,8],[2,3],[4,8],[4,3],[5,8],[5,3],[6,8],[6,3],[7,8],[7,3],[9,8],[9,3],[11,8],[11,3]]

    print(Solution().findJudge(3, trust))
    # print(Solution().findJudge2(3, trust))
```

## BFS and DFS LC 102、LC 104、LC 111.

广度优先搜索(Breadth-First-Search)

深度优先搜索(Depth-First-Search)

![](w12-kp-4.png)

在树或图中寻找特定节点

![](w12-kp-5.png)

![](w12-kp-6.png)

```
def BFS(graph, start, end):
    queue = []
    queue.append([start])
    visited.add(start)
    while queue:
        node = queue.pop()
        visited.add(node)
        process(node)
        nodes = generate_related_nodes(node)
        queue.push(nodes)
```

```
# DFS 代码递归写法
visited = set()
def dfs(node, visited):
    visited.add(node)
    # process current node here.
    for next_node in node.children():
        if not next_node in visited:
            dfs(next_node, visited)
```

```
# DFS 代码非递归写法
def DFS(self, tree):
    if tree.root is None:
        return []
    visited, stack = [], [tree.root]
    while stack:
        node = stack.pop()
        visited.add(node)
        process(node)
        nodes = generate_related_nodes(node)
        stack.push(nodes)
        # other processing work
```

### LC 102. Binary Tree Level Order Traversal 二叉树的层序遍历

Given the root of a binary tree, return the level order traversal of its nodes' values. (i.e., from left to right, level by level).

给你二叉树的根节点 root ，返回其节点值的 层序遍历 。 （即逐层地，从左到右访问所有节点）。

Example 1:

```
Input: root = [3,9,20,null,null,15,7]
Output: [[3],[9,20],[15,7]]
Example 2:
```

Example 2:

```
Input: root = [1]
Output: [[1]]
Example 3:
```

Example 3:

```
Input: root = []
Output: []
```

Constraints:

- The number of nodes in the tree is in the range [0, 2000]. (树中节点数目在范围 [0, 2000] 内)

- -1000 <= Node.val <= 1000

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

### LC 104. Maximum Depth of Binary Tree 二叉树的最大深度

Given the root of a binary tree, return its maximum depth.

A binary tree's maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.

给定一个二叉树，找出其最大深度。二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。

说明: 叶子节点是指没有子节点的节点。

Example 1:

```
给定二叉树 [3,9,20,null,null,15,7]，

    3
   / \
  9  20
    /  \
   15   7
返回它的最大深度 3 。

```

Example 2:

```
Input: root = [1,null,2]
Output: 2
```

Constraints:

- The number of nodes in the tree is in the range [0, $10^4$].
- -100 <= Node.val <= 100

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


### LC 111. Minimum Depth of Binary Tree 二叉树的最小深度

Given a binary tree, find its minimum depth.

The minimum depth is the number of nodes along the shortest path from the root node down to the nearest leaf node.

Note: A leaf is a node with no children.

给定一个二叉树，找出其最小深度。

最小深度是从根节点到最近叶子节点的最短路径上的节点数量。

说明：叶子节点是指没有子节点的节点。

Example 1:

```
Input: root = [3,9,20,null,null,15,7]
Output: 2
Example 2:
```

Example 2:

```
Input: root = [2,null,3,null,4,null,5,null,6]
Output: 5
```

Constraints:

- The number of nodes in the tree is in the range [0, $10^5$]. (树中节点数的范围在 [0, $10^5$] 内)

- -1000 <= Node.val <= 1000

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

## LC 787. Cheapest Flights Within K Stops, K 站中转内最便宜的航班 and LC 934. Shortest Bridge 最短的桥

## LC 787

There are n cities connected by some number of flights. You are given an array flights where flights[i] = [fromi, toi, pricei] indicates that there is a flight from city fromi to city toi with cost pricei.

You are also given three integers src, dst, and k, return the cheapest price from src to dst with at most k stops. If there is no such route, return -1.

有 n 个城市通过一些航班连接。给你一个数组 flights ，其中 flights[i] = [fromi, toi, pricei] ，表示该航班都从城市 fromi 开始，以价格 pricei 抵达 toi。

现在给定所有的城市和航班，以及出发城市 src 和目的地 dst，你的任务是找到出一条最多经过 k 站中转的路线，使得从 src 到 dst 的 价格最便宜 ，并返回该价格。 如果不存在这样的路线，则输出 -1。

Example 1:

```
Input: n = 4, flights = [[0,1,100],[1,2,100],[2,0,100],[1,3,600],[2,3,200]], src = 0, dst = 3, k = 1
Output: 700
Explanation:
The graph is shown above.
The optimal path with at most 1 stop from city 0 to 3 is marked in red and has cost 100 + 600 = 700.
Note that the path through cities [0,1,2,3] is cheaper but is invalid because it uses 2 stops.

```

Example 2:

```
Input: n = 3, flights = [[0,1,100],[1,2,100],[0,2,500]], src = 0, dst = 2, k = 1
Output: 200
Explanation:
The graph is shown above.
The optimal path with at most 1 stop from city 0 to 2 is marked in red and has cost 100 + 100 = 200.
```

Example 3:

```
Input: n = 3, flights = [[0,1,100],[1,2,100],[0,2,500]], src = 0, dst = 2, k = 0
Output: 500
Explanation:
The graph is shown above.
The optimal path with no stops from city 0 to 2 is marked in red and has cost 500.
```

示例 1：

```
输入: 
n = 3, edges = [[0,1,100],[1,2,100],[0,2,500]]
src = 0, dst = 2, k = 1
输出: 200
解释: 
城市航班图如下
从城市 0 到城市 2 在 1 站中转以内的最便宜价格是 200，如图中红色所示。
```

示例 2：

```
输入: 
n = 3, edges = [[0,1,100],[1,2,100],[0,2,500]]
src = 0, dst = 2, k = 0
输出: 500
解释: 
城市航班图如下
从城市 0 到城市 2 在 0 站中转以内的最便宜价格是 500，如图中蓝色所示。
```

Constraints:

- 1 <= n <= 100

- 0 <= flights.length <= (n * (n - 1) / 2)

- flights[i].length == 3

- 0 <= fromi, toi < n

- $from_i$ != $to_i$

- 1 <= pricei <= $10^4$

- There will not be any multiple flights between two cities.(航班没有重复，且不存在自环)

- 0 <= src, dst, k < n

- src != dst

二维动态规划，矩阵维护最小开销，传递 k + 1 次

![](w13-kp-1.png)

```
class Solution {
    public int findCheapestPrice(int n, int[][] flights, int src, int dst, int k) {
        int dist[][] = new int[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i == j) dist[i][j] = 0;
                else dist[i][j] = 0x3f3f3f3f;
            }
        }
        for (int ii = 0; ii <= k; ii++) {
            boolean changed = false;
            int dist1[][] = new int[n][n];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    dist1[i][j] = dist[i][j];
                    for (int kk = 0; kk < n; kk++) {
                        if (flights.length > kk && flights[kk].length > j && flights[kk][j] >= 0 && dist[i][kk] + flights[kk][j] < dist1[i][j]) {
                            changed = true;
                            dist1[i][j] = dist[i][kk] + flights[kk][j];
                        }
                    }
                }
            }
            dist = dist1;
            if (!changed) break;
        }
        return dist[src][dst] == 0x3f3f3f3f ? -1 : dist[src][dst];
    }
}
```

```
from typing import List
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

## LC 934

You are given an n x n binary matrix grid where 1 represents land and 0 represents water.

An island is a 4-directionally connected group of 1's not connected to any other 1's. There are exactly two islands in grid.

You may change 0's to 1's to connect the two islands to form one island.

Return the smallest number of 0's you must flip to connect the two islands.

在给定的二维二进制数组 A 中，存在两座岛。（岛是由四面相连的 1 形成的一个最大组。）

现在，我们可以将 0 变为 1，以使两座岛连接起来，变成一座岛。

返回必须翻转的 0 的最小数目。（可以保证答案至少是 1 。）

Example 1:

```
Input: grid = [[0,1],[1,0]]
Output: 1
```

Example 2:

```
Input: grid = [[0,1,0],[0,0,0],[0,0,1]]
Output: 2
```

Example 3:

```
Input: grid = [[1,1,1,1,1],[1,0,0,0,1],[1,0,1,0,1],[1,0,0,0,1],[1,1,1,1,1]]
Output: 1
```

Constraints:

- n == grid.length == grid[i].length

- 2 <= n <= 100

- grid[i][j] is either 0 or 1.

- There are exactly two islands in grid.

```
/* 思路: DFS + BFS */
/* 先用深度优先搜索DFS, 找到第1个岛屿, 将岛屿元素置为2, 并入队   */
/* 再用广度优先搜索BFS, 从第1个岛屿元素开始向外寻找, 找到的0置为2 */
/* 当找到第一个1时, 就返回寻找的路径step                       */

/* 队列结构体定义 */
typedef struct {
    int x;
    int y;
    int step;
} Queue;

/* DFS 寻找第一个岛屿元素 */
void dfs(int **A, int ASize, int i, int j, Queue *Q, int *rear) {
    if (i < 0 || i >= ASize || j < 0 || j >= ASize || A[i][j] != 1) {
        return;
    }
    /* 元素置为2, 并入队, step置为0 */
    A[i][j]           = 2;
    Q[(*rear)].x      = i;
    Q[(*rear)].y      = j;
    Q[(*rear)++].step = 0;

    /* 上下左右继续寻找 */
    dfs(A, ASize, i - 1, j, Q, rear); 
    dfs(A, ASize, i + 1, j, Q, rear);
    dfs(A, ASize, i, j - 1, Q, rear);
    dfs(A, ASize, i, j + 1, Q, rear);
    return;
}

int shortestBridge(int** A, int ASize, int* AColSize){
    Queue *Q = (Queue*)malloc(sizeof(Queue) * ASize * ASize);
    int front = 0;
    int rear  = 0;
    int find  = 0;
    int i, j, x, y, xx, yy, step;
    int xShift[] = {-1, 1,  0, 0};
    int yShift[] = { 0, 0, -1, 1};

    /* DFS第一个岛屿 */
    for (i = 0; i < ASize; i++) {
        for (j = 0; j < ASize; j++) {
            if (A[i][j] == 1) {
                dfs(A, ASize, i, j, Q, &rear);
                find = 1;
                break;
            }
        }
        /* 只寻找第一个岛屿 */
        if (find == 1) {
            break;
        }
    }

    /* BFS 第一个岛屿向外扩散 */
    while (front != rear) {
        x    = Q[front].x;
        y    = Q[front].y;
        step = Q[front++].step;

        /* 上下左右扩散 */
        for (i = 0; i < 4; i++) {
            xx = x + xShift[i];
            yy = y + yShift[i];
            if (xx < 0 || xx >= ASize || yy < 0 || yy >= ASize || A[xx][yy] == 2) {
                continue;
            }
            if (A[xx][yy] == 1) { /* 找到另一岛屿时, 返回step */
                return step;
            }
            A[xx][yy]      = 2; /* 将扩散到的0置为2, 并入队 */
            Q[rear].x      = xx;
            Q[rear].y      = yy;
            Q[rear++].step = step + 1;
        }
    }
    free(Q);
    return step;
}
```

把一坨 1 全标为 2 作为区分，然后扩散2的这一坨，看看扩散几次后会到 1 的位置

![](w13-kp-2.png)

```
class Solution {
    int[][] grid;
    public void mark2(int i, int j) {
        grid[i][j] = 2;
        if (i > 0 && grid[i-1][j] == 1) mark2(i-1, j);
        if (j > 0 && grid[i][j-1] == 1) mark2(i, j-1);
        if (i < grid.length-1 && grid[i+1][j] == 1) mark2(i+1, j);
        if (j < grid[i].length-1 && grid[i][j+1] == 1) mark2(i, j+1);
    }
    public boolean spread2() {
        int grid1[][] = new int[grid.length][grid[0].length];
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[i].length; j++) {
                if (grid[i][j] == 2) {
                    grid1[i][j] = 2;
                    if (i > 0) grid1[i-1][j] = 2;
                    if (j > 0) grid1[i][j-1] = 2;
                    if (i < grid.length-1) grid1[i+1][j] = 2;
                    if (j < grid[i].length-1) grid1[i][j+1] = 2;
                }
            }
        }
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[i].length; j++) {
                if (grid[i][j] == 1) {
                    if (grid1[i][j] == 2) return true;
                    grid1[i][j] = 1;
                }
            }
        }
        grid = grid1;
        return false;
    }
    public int shortestBridge(int[][] grid) {
        this.grid = grid;
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[i].length; j++) {
                if (grid[i][j] == 1) {
                    mark2(i, j);
                    int count = 0;
                    while (!spread2()) count++;
                    return count;
                }
            }
        }
        return -1;
    }
}
```

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

## 最短路径问题


1. 求解最短路径问题算法

- 狄克斯特拉（Dijkstra）算法

- 贝尔曼-福特（Bellman-Ford）算法

2. 最短路径问题

- 非加权图（unweighted graph） -> 广度优先搜索

- 加权图（weightedgraph） -> 狄克斯特拉算法; 贝尔曼-福特算法

![](w13-kp-3.png)

![](w13-kp-4.png)

![](w13-kp-5-2.png)

## 狄克斯特拉（Dijkstra）算法

狄克斯特拉 (Dijkstra) 算法是一种求解最短路径问题的算法，使用它可以求得从起点到终点的路径中权重总和最小的路径。

狄克斯特拉算法的名称取自该算法的提出者埃德斯加·狄克斯特拉，他在 1972 年获得了图灵奖。

1. 设 A 为起点、G 为终点

2. 首先设置各个顶点的权重起点为,其他顶点为无穷大。

3. 用红色表示目前所在的顶点。

4. 用绿色表示候补顶点。

5. 计算各个候补顶点的权重。

7. 从候补顶点中选出权重最小的顶点。

8. 确定了最短路径，移动到顶点 B。

11. 更新了剩下的顶点 D 和 E。

12. 到达终点 G ,搜索结束。最终得到的这颗橙色的树就是最短路径树，它表示了起点到达各个顶点的最短路径。

### 算法包含 4 个步骤 : 

1. 找出“最便宜”的节点，即可在最短时间内到达的节点。

2. 更新该节点的邻居的开销。

3. 重复这个过程，直到对图中的每个节点都这样做。

4. 计算最终路径。

![](w13-kp-6.png)

```
nodes = ('S', 'A', 'B', 'E')
distances = {
    'S': {'A': 6, 'B': 2},
    'A': {'E': 1},
    'B': {'A': 3, 'E': 5},
    'E': {'E': 0}}

unvisited = {node: None for node in nodes} #把None作为无穷大使用
visited = {}   #用来记录已经松弛过的数组
current = 'S'  #要找B点到其他点的距离
currentDistance = 0
unvisited[current] = currentDistance  #B到B的距离记为0
while True:
    print(current)
    for neighbour, distance in distances[current].items():
        if neighbour not in unvisited: continue   #被访问过了，跳出本次循环
        newDistance = currentDistance + distance  #新的距离
        if unvisited[neighbour] is None or unvisited[neighbour] > newDistance: #如果两个点之间的距离之前是无穷大或者新距离小于原来的距离
            unvisited[neighbour] = newDistance  #更新距离
    visited[current] = currentDistance  #这个点已经松弛过，记录
    del unvisited[current]  #从未访问过的字典中将这个点删除
    if not unvisited: break  #如果所有点都松弛过，跳出此次循环
    candidates = [node for node in unvisited.items() if node[1]]  #找出目前还有哪些点未松弛过
    current, currentDistance = sorted(candidates, key=lambda x: x[1])[0]  #找出目前可以用来松弛的点
    print(visited)
print(visited)
```

![](w13-kp-7.png)

### 狄克斯特拉 (Dijkstra) 算法复杂度

将图的顶点数设为 n、边数设为 m，那么如果事先不进行任何处理，该算法的时间复杂度就是 $O(n^2)$ 。不过，如果对数据结构进行优化，那么时间复杂度就会变为 O(m + nlogn)。

![](w13-kp-4.png)

![](w13-kp-8.png)

## 狄克斯特拉（Dijkstra）算法失效

![](w13-kp-9.png)

如果图中含有负数权重，狄克斯特拉算法可能会无法得出正确答案。

```
nodes = ('A', 'B', 'C', 'G')
distances = {
    'A': {'B': 2, 'C': 4},
    'B': {'A': 1, 'G': 1},
    'C': {'G': 1, 'B': -3},
    'G': {'G': 0}}
unvisited = {node: None for node in nodes} #把None作为无穷大使用
visited = {}   #用来记录已经松弛过的数组
current = 'A'  #要找B点到其他点的距离
currentDistance = 0
unvisited[current] = currentDistance  #B到B的距离记为0

while True:
    print(current)
    for neighbour, distance in distances[current].items():
        if neighbour not in unvisited: continue   #被访问过了，跳出本次循环
        newDistance = currentDistance + distance  #新的距离
        if unvisited[neighbour] is None or unvisited[neighbour] > newDistance: #如果两个点之间的距离之前是无穷大或者新距离小于原来的距离
            unvisited[neighbour] = newDistance  #更新距离
    visited[current] = currentDistance  #这个点已经松弛过，记录
    del unvisited[current]  #从未访问过的字典中将这个点删除
    if not unvisited: break  #如果所有点都松弛过，跳出此次循环
    candidates = [node for node in unvisited.items() if node[1]]  #找出目前还有哪些点未松弛过
    current, currentDistance = sorted(candidates, key=lambda x: x[1])[0]  #找出目前可以用来松弛的点
    print(visited)
print(visited)
```


## 贝尔曼-福特（Bellman-Ford）算法


1. 贝尔曼-福特 (Bellman-Ford) 算法是一种在图中求解最短路径问题的算法。该算法可以处理包含负值权重的图。

2. 贝尔曼- 福特算法的名称取自其创始人理查德·贝尔曼和莱斯特·福特的名字。贝尔曼也因为提出了该算法中的一个重要分类“动态规划”而被世人所熟知。

![](w13-kp-10.png)

![](w13-kp-11.png)

1. 设 A 为起点、G 为终点

2. 初始化

3. 用绿色表示被选中的候补顶点。

5. 用橙色表示路径。

7. 对所有的边都执行同样的操作。

8. 数值更新了,顶点C的权重变成了 2。

9. 同样地,再选出一条边・ ......

10. 此处顶点B因为边 B-C 而更新了权重，所以路径从之前的 A-B 变为了现在的 B-C。因此 A-B 不再以橙色表示，而 B-C 变为橙色。

12. 更新边 B-D 和边 B-E。



13. 更新边 C-D 和边 C-F。

14. 更新完所有的边后，第 1 轮更新就结束了。

15. 第 2 轮更新也结束了。顶点 B 的权重从 8 变成了 7，顶点 E 的权重从 9 变成了 8。接着，再执行一次更新操作。

16. 第 3 轮更新结束，所有顶点的权重都不再更新，操作到此为止。算法的搜索流程也就此结束，我们找到了从起点到其余各个顶点的最短路径。

17. 根据搜索结果可知，从起点 A 到终点 G 的最短路径是 A-C-D-F-G，权重为 14。

### 贝尔曼-福特 (Bellman-Ford) 算法复杂度

将图的顶点数设为 n 、边数设为 m，该算法经过 n 轮更新操作后就会停止，而在每轮更新操作中都需要对各个边进行1次确认，因此1轮更新所花费的时间就是 O(m) ，整体的时间复杂度就是 O(nm)。

![](w13-kp-12.png)

### 含有负值权重的最短路径问题

![](w13-kp-13.png)

## 算法复杂度

1. 时间复杂度

- 最坏情况下的时间复杂度 W(n)

- 平均情况下的时间复杂度 A(n)

2. 空间复杂度

### 插入排序

![w14-kp-1.png](attachment:w14-kp-1.png)

```
def Sort(alist):
    for index in range(1,len(alist)):
        currentvalue = alist[index]
        position = index
        while position>0 and alist[position-1]>currentvalue:
            alist[position]=alist[position-1]
            position = position-1
        alist[position]=currentvalue
```

```
def insertionSort(alist):
    for index in range(1,len(alist)):
        print(index)
        currentvalue = alist[index]
        position = index
        while position>0 and alist[position-1]>currentvalue:
            alist[position]=alist[position-1]
            position = position-1
        alist[position]=currentvalue
        print(alist)
print("insertionSort:")
alist = [5,7,1,3,6,2,4]
print(alist)
insertionSort(alist)
print(alist)
```

### 检索 & 顺序检索 & 平均情况时间估计

1. 检索

输入: 非降顺序排列的数组 L,元素数n, 数 x

输出: j 若 x 在 L 中, j 是 x 首次出现的下标; 否则 j = 0

基本运算: x 与 L 中元素的比较

2. 顺序检索

j = 1, 将 x 与 L[j] 比较. 如果 x = L[j], 则算法停止,输出 j;

如果不等 , 则把 j 加 1, 继续 x 与 L[j] 的比较, 如果 j >  n , 则停机并输出 0.

实例 [ [1] [2] [3] [4] [5] ]

x = 4 ,需要比较 4 次

x = 2.5 ,需要比较 5 次

3. 平均情况时间估计

输入实例的概率分布:

假设 x 在 L 中概率是 p ,且每个位置概率相等

$$
\begin{aligned}
A(n) &=\sum_{i=1}^{n} i \frac{p}{n}+(1-p) n \\
&=\frac{p(n+1)}{2}+(1-p) n
\end{aligned}
$$
当 $p = 1 / 2$ 时,
$$
A(n)=\frac{n+1}{4}+\frac{n}{2} \approx \frac{3 n}{4}
$$


### 函数的渐近的界

以元素比较作基本运算


五种表示函数的阶的符号 : $O, \Omega, o, \omega, \theta$

EX : $T(n)=n^{2}+8 n-5$

#### 大 O 符号

定义:设 $f$ 和 $g$ 是定义域为自然数集 N 上的函数.

若存在正数 c 和 $n_{0}$,使得对一切 $n \geq n_{0} $ 有

$0 \leq f(n) \leq c g(n)$

成立,则称 f(n) 的漸近的上界是 g(n), 记作

$f(n) = O(g(n))$

设 $f(n) = n^2 + n$, 则

$f(n)= O(n^2)$, 取 $c=2, n_0=1$ 即可

$f(n)= O(n^3)$, 取 $c=1, n_0=2$ 即可

1. $f(n)=O(g(n))$, f(n) 的阶不高于 g(n) 的阶.

2. 可能存在多个正数 c ,只要指出一个即可.

3. 对前面有限个值可以不满足不等式.

4. 常函数可以写作 O(1).

#### 大 $\Omega$ 符号

定义:设 f 和 g 是定义域为自然数集 N 上的函数. 

若存在正数 c 和 $n_0$, 使得对一切 $n \geq n_{0}$ 有

$0 \leq cg(n) \leq f(n)$

成立, 则称 f(n) 的渐近的下界是 g(n), 记作

$f(n)=\Omega(g(n))$

设 $f(n) = n^2 + n$ , 则

$f(n)=\Omega(n^{2})$, 取 $c = 1, n_{0} = 1$ 即可

$f(n)=\Omega(100n)$ , 取 $c = 1/100, n_{0} = 1$ 即可

1. $f(n) = \Omega(g(n)), f(n)$ 的阶不低于 g(n) 的阶.

2. 可能存在多个正数 c,指出一个即可

3. 对前面有限个 n 值可以不满足上述不等式.

#### 小 o 符号

定义: 设 f 和 g 是定义域为自然数集 N 上的函数.

若对于任意正数 c 都存在 $n_{0}$, 使得对一切 $n \geq n_{0}$ 有

$0 \leq f(n) \leq c g(n)$

成立, 则记作

$f(n)=o(g(n))$
例子: $f(n) = n^{2} + n$, 则

$f(n) = o(n^{3})$

$c \geq 1$ 显然成立, 因为 $n^{2} + n < cn^{3} (n_{0} = 2)$

任给 $ 1 > c > 0$, 取 $n_{0} >\lceil 2 / c\rceil$ 即可。

因为 $cn \geq cn_{0} >2$ (当 $n \geq n_0$)

$n^{2} + n < 2 n^{2} < cn^{3}$

1. $f(n)=o(g(n)), f(n)$ 的阶低于 $g(n)$ 的阶

2. 对不同正数 $c, n_{0}$ 不一样. c 越小 $n_{0}$ 越大.

3. 对前面有限个 n 值可以不满足不等式.


### 小 $\omega$ 符号

定义: 设 f 和 g 是定义域为自然数集 N 上的函数。

若对于任意正数 c 都存在 $n_{0}$。, 使得对一切 $n \geq n_{0}$ 有

$$0 \leq \operatorname{cg}(n) < f(n)$$

成立, 则记作

$f(n)= \omega(g(n))$

设 $f(n) = n^{2} + n$ , 则 $f(n) = \omega(n)$,
不能写 $f(n)=\omega(n^{2})$, 因为取 $c = 2$, 不存在 $n_{0}$ 使得对一切 $n \geq n_{0}$ 有下式成立

$c n^{2}=2 n^{2} < n^{2}+n $ (錯誤示範)

1. $f(n)=\omega(g(n)), f(n)$ 的阶高于 $g(n)$ 的阶.

2. 对不同的正数 $c$, $n_{0}$ 不等, $c$ 越大 $n_{0}$ 越大.

3. 对前面有限个 n 值可以不满足不等式.

#### $\Theta$ 符号

若 $f(n)=O(g(n))$ 且 $f(n)=\Omega(g(n))$, 则记作

$f(n)=\Theta(g(n))$

例子: $f(n)=n^{2} + n$ , g(n) = 10 0n^{2}, 那么有

$f(n)=\Theta(g(n))$

1.$f(n)$ 的阶与 $g(n)$ 的阶相等。

2.对前面有限个 $n$ 值可以不满足条件.


#### 例子 : 素数测试

算法 Primality Test(n)
输入: n,大于 2 的奇整数
输出: true 或者 false

 $\left.s \leftarrow \lfloor n^{1 / 2}\right\rfloor$
 
 for $j \leftarrow 2$ to s
 
     if j 整除 n
     
     then return false
     
 return true

问题:

若 $n^{1/2}$ 可在 $O(1)$ 计算 , 基本运算是整除 , 以下表示是否正确?

$W(n)=O(n^{1/2})$(正確)

$W(n)=\Theta(n^{1/2})$(錯誤)

Why!?

#### 有关函数渐近的界的定理

![](attachment:w14-kp-3.png)

![](attachment:w14-kp-4.png)

![](attachment:w14-kp-5.png)

![](attachment:w14-kp-6.png)

![](attachment:w14-kp-7.png)

![](attachment:w14-kp-8.png)

![](attachment:w14-kp-9.png)

![](attachment:w14-kp-10.png)

#### 小结

1. 估计函数的阶的方法：

- 计算极限

- 阶具有传递性

2. 对数函数的阶低于幂函数的阶，多项式函数的阶低于指数函数的阶。

3. 算法的时间复杂度是各步操作时间之和，在常数步的情况下取最高阶的函数即可。

## 基本函数类

![](w14-kp-11.png)


### 对数函数

![](w14-kp-12.png)

### 有关性质的说明

![](w14-kp-13.png)

### 指数函数与阶乘

![](w14-kp-14.png)

### 应用：估计搜索空间大小

![](w14-kp-15.png)

### $\log (n !) ?(n \log n)$

$\log (n !)=\Theta(n \log n)$

![](w14-kp-16.png)

### 取整函数 & 性質

![](w14-kp-17.png)

![](w14-kp-18.png)

### 按照阶排序

![](w14-kp-19.png)

### 序列求和的方法

![](w14-kp-20.png)

### 二分检索算法 & 运行实例 & 平均时间复杂度

![](w14-kp-21.png)

### 估计和式上界的放大法 & 举例 & 估计和式渐近的界

![](w14-kp-22.png)

### 小結

1. 序列求和基本公式:

- 等差数列

- 等比数列

- 调和级数

2. 估计序列和:

- 放大法求上界

- 用积分做和式的渐近的界

3. 应用: 计数循环过程的基本运算次数

## 递推方程与算法分析

![](w14-kp-23.png)

## 迭代法求解递推方程

- 不断用递推方程的右部替换左部

- 每次替换，随着n的降低在和式中多出一项

- 直到出现初值停止迭代

- 将初值代入并对和式求和

- 可用数学归纳法验证解的正确性

![](w14-kp-24.png)

![](w14-kp-25.png)

![](w14-kp-26.png)

![](w14-kp-27.png)

## 蒙地卡羅

### 1. Monte Carlo

The term "Monte Carlo method" was firstly introduced in 1947 by Nicholas Metropolis.

### 2. Reference

Metropolis. The beginning of the Monte Carlo method. Los Alamos Science, 125–130, 1987.

1. MIT 6.0002 Introduction to Computational Thinking and Data - Monte Carlo Simulation : https://www.youtube.com/watch?v=OgO1gpXSUzU

2. 蒙特卡洛原理代码 monte carlo : https://blog.csdn.net/yjinyyzyq/article/details/86600393

3. 蒙特卡洛法高维数值积分： Vegas : https://zhuanlan.zhihu.com/p/264315872

4. 一文详解蒙特卡洛（Monte Carlo）法及其应用 : https://blog.csdn.net/qq_39521554/article/details/79046646

### 3. Monte Carlo Algorithms

Monte Carlo refers to algorithms that rely on repeated random sampling to obtain numerical results.

The output of Monte Carlo algorithms can be incorrect.

 - In all of our examples, the algorithms' outputs are incorrect.

 - But they are close to the correct solution.


### 4. Application 1: Calculating Pi

We already know $\pi \approx 3.141592653589$ ...

Pretend we do not know the value of $\pi$.

Can we find it out (approximately) using a random number generator?

![](w15-kp-1.png)

### 5. Application 2: Buffon's Needle Problem

![](w15-kp-2.png)

### 6. Application 3: Area of A Region

![](w15-kp-3.png)

### 7. Application 4: Integration

- Integration

- Monte Carlo Integration (Univariate)

- Monte Carlo Integration (Univariate): Example

- Monte Carlo Integration (Multivariate)

- Monte Carlo Integration: Bivariate Example

![](w15-kp-4.png)

![](w15-kp-5.png)

### 8. Application 5: Estimate of Expectation

![](w15-kp-6.png)


## 蒙地卡羅 & PI

### M1

![](w16-kp-1.png)

### M2

蒙特卡洛分析法（Monte Carlo method）（统计模拟法），是一种采用随机抽样（Random Sampling）统计来估算结果的计算方法，可用于估算圆周率，由约翰·冯·诺伊曼提出。由于计算结果的精确度很大程度上取决于抽取样本的数量，一般需要大量的样本数据，因此在没有计算机的时代并没有受到重视。

蒙特卡洛分析法利用正方形的面积跟正方形体内最大圆的面积比值为4：π，来求解 π 的值，假设给正方形内部随机仍石子，那么落在圆里面的概率为四分之π，当石子数量足够大的时候就可以精确的求出 π 的值

```
import random #引入随机数库
N = 1000*1000 #定义循环次数
list=0.0 #落入圆中的石子数量
for i in range(1,1+N):
    x = random.uniform(0,1) #模拟石子的随机坐标
    y = random.uniform(0,1)
    if (pow(x,2) + pow(y,2)) <= 1:
        list = list+1
print(4*(list/N)) #打出圆周率
```

### M2 结论

1、实验条件相同的情况下，每次结果并不一致，都有波动

2、实验条件不同的情况下，数据量越多，则每次结果的波动越小

3、实验条件不同的情况下，数据量越多，实验结果距离真实值的误差越小

### M3

由 Monte Carlo 方法估计出圆面积 S，再由 S = PI * r_{2} 计算出 PI

在边长 2r 的正方形中随机撒 N 个点，如果有 M 个落在内切圆中，则圆与正方形面积之比近似为M:N

$PI = 4 * \frac{M}{N}$

![](w16-kp-2.png)

### 伪随机数

- 由确定性的算法计算出的随机数序列，循环周期极长

- 并非真正的随机数，计算时的初值不变，则结果数序也不变

- 但有着类似于随机数的统计特征，如均匀性、独立性等

### 生成方法

- 线性同余

- 平方取中

- 量子随机发生器

### 伪随机数质量评价

Bundesamt für Sicherheit in der Informationstechnik

- A sequence of random numbers with a low probability of containing identical consecutive elements.

- A sequence of numbers which is indistinguishable from 'true random' numbers according to specified statistical tests.

- It should be impossible for any attacker (for all practical purposes) to calculate, or otherwise guess, from any given sub-sequence, any previous or future values in the sequence, nor any inner state of the generator.

- It should be impossible, for all practical purposes, for an attacker to calculate, or guess from an inner state of the generator, any previous numbers in the sequence or any previous inner generator states.

### Python 中的伪随机数

MT19937

Mersenne Twister: 周期长度通常取 Mersenne 质数

速度快，周期长(可达 $2^{19937} − 1$)， 623 维均匀分布

![](w16-kp-3.png)

### M4

![](w15-lab-2.gif)

```
import javax.swing.*;
import javax.swing.event.*;
import java.awt.*;
import java.awt.event.*;
import java.util.Random;
public class Mondecaro extends JFrame implements ActionListener {
    private JButton button;
    private JPanel panel;
    private int height = 300, width = 300;
    public static void main (String[] argv) {
        Mondecaro frame = new Mondecaro();
        frame.setSize(520, 350);
        frame.createGUI();
        frame.setVisible(true);
    }
    private void createGUI() {
        setDefaultCloseOperation(EXIT_ON_CLOSE);
        Container window = getContentPane();
        window.setLayout(new FlowLayout());
        panel = new JPanel();
        panel.setPreferredSize(new Dimension(width, height));
        panel.setBackground(Color.white);
        window.add(panel);
        button = new JButton("Mondecaro");
        window.add(button);
        button.addActionListener(this);
    }
    public void actionPerformed(ActionEvent e) {
        draw();
    }
    private void draw() {
        Graphics paper = panel.getGraphics();
        paper.setColor(Color.white);
        paper.fillRect( 0, 0, width, height);
        paper.setColor(Color.black);
        int px = width, py = height / 2;
        Random random = new Random ();
        int n = width / 2, in = 0, N = 10000;
        for (int i = 1; i <= N; i++) {
            int x = random.nextInt(width);
            int y = random.nextInt(width);
            double dist = Math.sqrt(Math.pow(x - n, 2) + Math.pow(y - n, 2));
            if (dist < n) {
                paper.setColor(Color.red);
                in++;
            } else {
                paper.setColor(Color.blue);
            }
            paper.drawOval( x, y, 1, 1);
            setTitle("Area = " + (double)in / i * 4);
            for (int k = 0; k < 1000; k++); 
        }
    }
}
```
```
import random
total = [10, 100, 1000, 10000, 100000, 1000000, 5000000]  #随机点数
for t in total:
    in_count = 0
    for i in range(t):
        x = random.random()
        y = random.random()
        dis = (x**2 + y**2)**0.5
        if dis<=1:
            in_count += 1
    print(t,'个随机点时，π 是：', 4 * in_count/t)
```

蒙特卡罗方法是一种计算方法。原理是通过大量随机样本，去了解一个系统，进而得到所要计算的值。
 
它非常强大和灵活，又相当简单易懂，很容易实现。对于许多问题来说，它往往是最简单的计算方法，有时甚至是唯一可行的方法。它诞生于上个世纪 40 年代美国的"曼哈顿计划"，名字来源于赌城蒙特卡罗，象征概率。

通常蒙特卡罗方法可以粗略地分成两类：

一类是所求解的问题本身具有内在的随机性，借助计算机的运算能力可以直接模拟这种随机的过程。例如在核物理研究中，分析中子在反应堆中的传输过程。中子与原子核作用受到量子力学规律的制约，人们只能知道它们相互作用发生的概率，却无法准确获得中子与原子核作用时的位置以及裂变产生的新中子的行进速率和方向。科学家依据其概率进行随机抽样得到裂变位置、速度和方向，这样模拟大量中子的行为后，经过统计就能获得中子传输的范围，作为反应堆设计的依据。

另一种类型是所求解问题可以转化为某种随机分布的特征数，比如随机事件出现的概率，或者随机变量的期望值。通过随机抽样的方法，以随机事件出现的频率估计其概率，或者以抽样的数字特征估算随机变量的数字特征，并将其作为问题的解。这种方法多用于求解复杂的多维积分问题。

### π 的计算

第一个例子是，如何用蒙特卡罗方法计算圆周率π。正方形内部有一个相切的圆，它们的面积之比是 π/4

现在，在这个正方形内部，随机产生 10000 个点（即 10000 个坐标对 (x, y)），计算它们与中心点的距离，从而判断是否落在圆的内部。

如果这些点均匀分布，那么圆内的点应该占到所有点的 π/4，因此将这个比值乘以4，就是π的值。通过R语言脚本随机模拟30000个点，π的估算值与真实值相差0.07%。

```
import numpy as np
import tqdm
#random_generate = np.random.uniform(low=0.0, high=2.0, size=(1, 1))

#求解 pi
sum = 0
for i in tqdm.tqdm(range(3000000)):
        #random_generate = np.random.rand(2
        random_generate = np.random.uniform(low=0.0, high=2.0, size=(2))
        if np.sum(np.square(random_generate-np.array([1.0, 1.0]))) <=1:
                sum += 1
print(sum)
pi = 4 * (sum / 3000000)
print('pi is:{}'.format(pi))
```
### 蒙特卡罗方法求定积分

比如积分 $\theta=\int_{a}^{b} f(x) d x$ ，如果f(x)的原函数很难求解，那么这个积分也会很难求解。

而通过蒙特卡罗方法对其进行模拟求解的方式有二。

1. 随机投点法

这个方法和上面的两个例子的方法是相同的。如图所示，有一个函数f(x)，要求它从a到b的定积分，其实就是求曲线下方的面积：

这时可以用一个比较容易算得面积的矩型罩在函数的积分区间上（假设其面积为 Area），然后随机地向这个矩形框里面投点，其中落在函数f(x)下方的点为绿色，其它点为红色，然后统计绿色点的数量占所有点（红色+绿色）数量的比例为r，那么就可以据此估算出函数f(x)从 a 到 b 的定积分为 Area × r。

![](w15-lab-4.png)

```
# 求解定积分 x^2 区间[1, 2]; 投点法
import numpy as np
import tqdm
sum = 0
for i in tqdm.tqdm(range(3000000)):

        random_generate = np.array([np.random.uniform(1, 2), np.random.uniform(0, 4)])
        if np.square(random_generate[0]) > random_generate[1]:
                sum += 1
print(sum)
area = 4 * sum / 3000000
print('Area is:{}'.format(area))
```

2. 平均值法 (期望法)

如下图所示，在 [a,b] 之间随机取一点 x 时，它对应的函数值就是 f(x)，我们要计算 $\theta=\int_{a}^{b} f(x) d x$，就是图中阴影部分的面积。

![](w15-lab-5.png)

一个简单的近似求解方法就是用 $f(x) *(b-a)$ 来粗略估计曲线下方的面积，在 [a,b] 之间随机取点 x，用 f(x) 代表在 [a,b] 上所有 f(x) 的值，如下图所示：

![](w15-lab-6.png)

用一个值代表 [a,b] 区间上所有的 ( ) 的值太粗糙了，我们可以进一步抽样更多的点，比如下图抽样了四个随机样本 $x_{1}, x_{2}, x_{3}, x_{4} $ (满足均匀分布)，每个样本都能求出一个近似面积值 $f(x_{i}) * (b - a)$ ，然后计算他们的数学期望，就是蒙特卡罗计算积分的平均值法了。

![](w15-lab-7.png)

用数学公式表述上述过程：

$S = \frac{1}{4} [f(x_{1})(b - a) + f(x_{2})(b - a) + + f(x_{3})(b - a) + + f(x_{4})(b - a)] = \frac{1}{4} (b - a)(f(x_{1}) + f(x_{2}) + f(x_{3}) + f(x_{4})) = \frac{1}{4} (b - a)\sum_{i=1}^{4} f(x_{i}) $

然后进一步我们采样 n 个随机样本 (满足均匀分布)，则有：

$S = \frac{b - a}{n} \sum_{i=1}^{n} f(x_{i})\simeq \theta$

采样点越多，估计值也就越来越接近。

上面的方法是假定 x 在 [a,b] 间是均匀分布的，而大多时候 x 在 [a,b] 上不是均匀分布的，因此上面方法就会存在很大的误差。

这时我们假设x在[a,b]上的概率密度函数为 $p(x)$ ，加入到 $\theta=\int_{a}^{b} f(x) d x$ 中变换为：

 $\theta=\int_{a}^{b} f(x) d x = \int_{a}^{b} \frac{f(x)}{p(x)} p(x) d x \simeq \frac{1}{n} \sum_{i=1}^{n} \frac{f(x_{i})}{p(x_{i})}$

这就是蒙特卡罗期望法计算积分的一般形式。那么问题就换成了如何从 $p(x)$ 中进行采样。

```
# 求解定积分 X^2 区间 [1， 2]; 平均法
import numpy as np
import tqdm
sum = 0
for i in tqdm.tqdm(range(3000000)):
        random_x = np.random.uniform(1, 2, size=None)
        # None 是默认的也可以不写
        a =  np.square(random_x)
        sum += a*(2-1)
area = sum/3000000
print('calculate by mean_average:{}'.format(area))
```

### 蒙特卡洛法高维数值积分 Vegas

高能物理研究经常要用到高维函数的数值积分。传统的数值积分方法，比如梯形公式，辛普森积分，Gauss Quadrature 已经统统失效。原因很简单，那些算法每个维度需要至少 M 个离散的数据点，对于 N 维积分，需要在 $M^{N}$ 个点上计算函数取值。

比如 10 维积分，每个维度用最节省的 15 个点的 Gauss Quadrature，需要计算的函数值的次数也达到了 $M^{N} = 15^{10} = 576650390625$ 约 5766 亿次。

出现这种情况一般称作维数灾难。

在此使用蒙特卡洛积分算法 Vegas 做高维数值积分，而 Python 的 Vegas 库的安装以及 Vegas 蒙卡积分的原理如下。

1. VEGAS 高维函数蒙特卡洛积分

安装 vegas 库很简单，在命令行使用如下命令

```
pip install vegas
```

计算如下 4 维高斯函数

$f(x_{0}, x_{1}, x_{2}, x_{3}) = N exp( - \sum_{i=0}^{3} (x_{i} - \frac{1}{2})^{2} / 0.01 )$

在闭区间 $x_{0} \in [-1, 1], x_{1} \in [0, 1], x_{2} \in [0, 1], x_{3} \in [0, 1]$

上的数值积分。其中 N = 1013.211 是一个归一化因子。

```
# copy and paste to test.py
import vegas
import math

def f(x):
    dx2 = 0
    for d in range(4):
        dx2 += (x[d] - 0.5) ** 2
    return math.exp(-dx2 * 100.) * 1013.2118364296088

integ = vegas.Integrator([[-1, 1], [0, 1], [0, 1], [0, 1]])

result = integ(f, nitn=10, neval=1000)
print(result.summary())
print('result = %s    Q = %.2f' % (result, result.Q))
```

2. 蒙特卡洛积分 - 重要抽样法

这里忽略黎曼积分的适用性以及勒贝格积分的优越性讨论，来自 Mathematica 关于黎曼求和、黎曼积分的例子

![](w15-lab-8.png)

黎曼积分 原理指导我们，为了求一维函数 f(x) 在闭区间 [a, b] 上的定积分，可以先把区间分成 n 份， $a$ < $x_{1}$ < $x_{2}$< $\cdots$ <$ x_{n-1}$ < $b$  , 其中格子大小为 $\Delta x_{i}=x_{i+1}-x_{i}$ 。

函数的积分近似等于小格子中任一处的函数值 $f\left(x_{i}^{*}\right)$ 乘以 $\Delta x_{i}$ ，并对所有格子求和。

$F \approx \sum_{i} f\left(x_{i}^{*}\right) \Delta x_{i}$

因此可以使用均匀分布抽样出的 $x_{i}$ 点上函数值 $f(x_{i})$ 乘以平均间距 $\frac{x_{max}-x_{min}}{N}$ 求和来近似黎曼积分。


$F=\int_{x_{\min }}^{x_{\max }} f(x) d x \approx \frac{x_{\max }-x_{\min }}{N} \sum_{i=1}^{N} f\left(x_{i}\right)$


如果是高维积分，只需要把右边的 $x_{max} - x_{min}$ 换成体积 $\mathcal{V}$ , 积分公式变为，

$F \approx \frac{\mathcal{V}}{N} \sum_{i}^{N} f\left(x_{i}\right) \approx \mathcal{V} \mathrm{E}\left[f\left(x_{i}\right)\right] $

其中 $\mathrm{E}[f]$ 表示 $f$ 在均匀分布下的期望值。

马文淦的《计算物理》中介绍，根据中心极限定理，因为 F 是大量随机变量 $f(x_{i})$ 的求和，它的值满足正态分布。

蒙卡积分的误差 $\propto \frac{\sigma\left(f_{x_{i}}\right)}{\sqrt{n}}$ ，因此有两种办法可以提高蒙特卡洛积分的精度。

第一种是多撒点，将撒点个数 $n$ 每增加 100 倍，蒙卡积分的误差就会减小为原来的十分之一。这个结论独立于积分的维数。

第二种是减小 $x_{i}$ 点上集合 $\left\{f\left(x_{i}\right)\right\}$ 的涨落 $\propto (f x_{i})$ 。


如果 $f(x) = c$ 是常数，则集合 $\left\{f\left(x_{i}\right)\right\}$ 的方差最小，为 $\sigma^{2}=\left\langle(f-\langle f\rangle)^{2}\right\rangle=0$ 。

当 $f(x)$ 偏离均匀分布，在局部有很尖的峰，则集合  $\left\{f\left(x_{i}\right)\right\}$ 的方差 (涨落) 就会比较大。

减小被积函数方差的方法是选择一个与 $f(x)$ 形式相近，但比较好抽样的函数 $g(x)$, 将积分写为，

$F=\int_{\mathcal{V}} \frac{f(x)}{g(x)} g(x) d x=\mathcal{V} \mathbb{E}_{g}\left[\frac{f(x)}{g(x)}\right] $

其中期望值 $\mathbb{E}_{g}$ 表示按照概率密度函数 $g(x)$ 抽样出一系列点 $x_{i}$ ，并使用这些点计算 $f(x)/g(x)$ 的均值，

$\frac{1}{N} \sum_{i}^{N} \frac{f\left(x_{i}\right)}{g\left(x_{i}\right)}$

此时，因为 $f(x) ~ g(x)$ ，被积函数 $f(x)/g(x) ~ 1$ 接近常数， $\left\{f\left(x_{i}\right) / g\left(x_{i}\right) \right\}$ 方差更小，从理论上降低蒙卡积分的误差。

与暴力增加 n 相比， $g(x)$ 函数的具体形式依赖于被积函数。

Vegas 积分就是要使用适配的方式，自动寻找 g(x)。


## 近似算法

![](w16-kp-4.png)

如何找出覆盖全美50个州的最小广播台集合呢？ 集合覆盖问题：$O(2^{n})$

假设每秒可计算10个子集，所需的时间将如下：

在获得精确解需要的时间太长时，可使用近似算法。判断近似算法优劣的标准如下：

- 速度有多快；

- 得到的近似解与最优解的接近程度。

贪婪算法可化解危机！使用下面的贪婪算法可得到非常接近的解：

1. 选出这样一个广播台，即它覆盖了最多的未覆盖州。即便这个广播台覆盖了一些已覆盖的州，也没有关系。

2. 重复第一步，直到覆盖了所有的州。

贪婪算法是不错的选择，它们不仅简单，而且通常运行速度很快。在这个例子中，贪婪算法的运行时间为 $O(n^{2})$ ，其中 n 为广播台数量。

```
states_needed = set(["mt", "wa", "or", "id", "nv", "ut", "ca", "az"])
stations = {}
stations["kone"] = set(["id", "nv", "ut"])
stations["ktwo"] = set(["wa", "id", "mt"])
stations["kthree"] = set(["or", "nv", "ca"])
stations["kfour"] = set(["nv", "ut"])
stations["kfive"] = set(["ca", "az"])

final_stations = set()

while states_needed:
    best_station = None
    states_covered = set()
    for station, states in stations.items():
        covered = states_needed & states
        if len(covered) > len(states_covered):
            best_station = station
            states_covered = covered
    states_needed -= states_covered
    final_stations.add(best_station)
print(final_stations)
```

### 多机调度问题

任给有穷的作业集 A 和 m 台相同的机器，作业 a 的处理时间为正整数 t(a),每一项作业可以在任一台机器上处理。

如何把作业分配给机器才能使完成所有作业的时间最短?

即，如何把 A 划分成 m 个不相交的子集 $A_{i}$，使得

\max \left\{\sum_{a \in A_{i}} t(a) \mid i=1,2, \cdots, m\right\}

最小?

### 例子

一个运行实例例如，3 台机器，8 项作业,处理时间依次为: 3,4,3,6,5,3,8,4

算法的解:{1,4} ，{2,6,7}，{3,5,8},

负载: 3+6=9，4+3+8=15，3+5+4=12

完成作业时间:15

### 贪心法 G-MPS

机器 i 的负载:已分配给 i 的作业处理时间之和。

算法 G-MPS:

1. 按输入的顺序分配作业

2. 把每一项作业分配给当前负载最小的机器

3. 若当前负载最小的机器有2台或2台以上，则分配给其中的任意一台(比如标号最小的一台)

一个运行实例例如，3台机器，8项作业,处理时间依次为:3,4,3,6,5,3,8,4

算法的解:{1,4} ，{2,6,7}，{3,5,8},

负载:3+6=9，4+3+8=15，3+5+4=12

完成作业时间:15

最优解处理时间为3,4,3,6,5,3,8,4

最优解:{1,3,4}，{2,5,6}，{7,8}

负载:3+3+6=12，4+5+3=12，8+4=12

完成作业时间:12

### 贪心法 G-MPS 性能

定理 对多机调度问题每个有 m 台机器的实例 l，则 $\mathrm{G}-\operatorname{MPS}(I) \leq\left(2-\frac{1}{m}\right) \mathrm{OPT}(I)$

证 (1)最大负载不小于单任务负载

$\mathrm{OPT}(I) \geq \max _{a \in A} t(a)$

(2)最大负载不小于平均负载

$\mathrm{OPT}(I) \geq \frac{1}{m} \sum_{a \in A} t(a)$

设机器 $M_{j}$ 的负载最大，记作 $t(M_{j})$。设 b 是最后被分配给机器 $M_{j}$的作业。由算法,在分配 b 时 $M_{j}$ 的负载最小，则有：

$t\left(M_{j}\right)-t(b) \leq \frac{1}{m}\left(\sum_{a \in A} t(a)-t(b)\right)$

$G-\operatorname{MPS}(I) =t\left(M_{j}\right) \leq \frac{1}{m}\left(\sum_{a \in A} t(a)-t(b)\right)+t(b) \leq \frac{1}{m} \sum_{a \in A} t(a)+\left(1-\frac{1}{m}\right) t(b) =\left(2-\frac{1}{m}\right) \mathrm{OPT}(I)$


### 一个紧实例

m 台机器， m(m-1)+1 项作业，前 m(m-1) 项作业处理时间都为 1，最后一项作业处理时间为 m。

算法解：前 m(m-1) 项作业均分给 m 台机器，每台 m-1 项，最后作业任给一台机器，G-MPS(I)= 2m-1。

最优解：前 m(m-1) 项作业均分给 m-1 台机器，每台m项，最后 1 项分给留下的机器， OPT(I)= m。

G-MPS 是 2-近似算法。

### 改进的贪心近似算法

递降贪心法DG-MPS:

按处理时间从大到小重新排列作业,然后运用 G-MPS。例如对上述紧实例得到最优解.

对另一个实例:3,4,3,6,5,3,8,4

重新排序8,6,5,4,4,3,3,3

负载为:8+3=11,6+4+3=13,5+4+3=12

### 近似比

![](w16-kp-5.png)