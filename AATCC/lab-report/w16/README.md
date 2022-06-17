# AATCC - 算法分析和复杂性理论 - Analysis of Algorithms and Theory of Computational Complexity

> 2101212850 干皓丞

PKU 2022 個人實驗報告作業


## Details

第 16 週課堂課堂複習與作業。

### 楼梯算法 (SD)

楼梯算法 (SD) 在思路上和 O(1) 算法有很大不同，它抛弃了动态优先级的概念。而采用了一种完全公平的思路。前任算法的主要复杂性来自动态优先级的计算，调度器根据平均睡眠时间和一些很难理解的经验公式来修正进程的优先级以及区分交互式进程。这样的代码很难阅读和维护。楼梯算法思路简单，但是实验证明它对应交互式进程的响应比其前任更好，而且极大地简化了代码。

和 O(1) 算法一样，楼梯算法也同样为每一个优先级维护一个进程列表，并将这些列表组织在 Active数组中。当选取下一个被调度进程时，SD 算法也同样从active数组中直接读取。与 O(1) 算法不同在于，当进程用完了自己的时间片后，并不是被移到 expire 数组中。而是被加入 active 数组的低一优先级列表中，即将其降低一个级别。不过请注意这里只是将该任务插入低一级优先级任务列表中，任务本身的优先级并没有改变。当时间片再次用完，任务被再次放入更低一级优先级任务队列中。就象一部楼梯，任务每次用完了自己的时间片之后就下一级楼梯。任务下到最低一级楼梯时，如果时间片再次用完，它会回到初始优先级的下一级任务队列中。

比如某进程的优先级为 1，当它到达最后一级台阶 140 后，再次用完时间片时将回到优先级为 2 的任务队列中，即第二级台阶。不过此时分配给该任务的 time_slice 将变成原来的 2 倍。比如原来该任务的时间片 time_slice 为 10ms，则现在变成了 20ms。基本的原则是，当任务下到楼梯底部时，再次用完时间片就回到上次下楼梯的起点的下一级台阶。并给予该任务相同于其最初分配的时间片。总结如下：设任务本身优先级为P，当它从第N级台阶开始下楼梯并到达底部后，将回到第 N+1 级台阶。并且赋予该任务 N+1 倍的时间片。

以上描述的是普通进程的调度算法，实时进程还是采用原来的调度策略，即 FIFO 或者 Round Robin。

楼梯算法能避免进程饥饿现象，高优先级的进程会最终和低优先级的进程竞争，使得低优先级进程最终获得执行机会。对于交互式应用，当进入睡眠状态时，与它同等优先级的其他进程将一步一步地走下楼梯，进入低优先级进程队列。当该交互式进程再次唤醒后，它还留在高处的楼梯台阶上，从而能更快地被调度器选中，加速了响应时间。

楼梯算法的优点：从实现角度看，SD基本上还是沿用了 O(1) 的整体框架，只是删除了 O(1) 调度器中动态修改优先级的复杂代码；还淘汰了 expire 数组，从而简化了代码。它最重要的意义在于证明了完全公平这个思想的可行性。

完全公平调度器 (英语：Completely Fair Scheduler，缩写为 CFS)，Linux 内核的一部分，负责进程调度。参考了 Con Kolivas 提出的调度器源代码后，由匈牙利程式员 Ingo Molnar 所提出。在 Linux kernel 2.6.23 之后采用，取代先前的 O(1) 调度器，成为系统预设的调度器。它负责将 CPU 资源，分配给正在执行中的进程，目标在于最大化程式互动效能与整体 CPU 的使用率。使用红黑树来实作，算法效率为 O(log(n))。

### 背景

CFS 是首支以公平伫列 (fair queuing) 的调度器可应用于一般用途操作系统 (general-purpose operating system).[1]

CFS调度器参考了康恩·科里瓦斯 (Con Kolivas) 所开发的楼梯调度算法（staircase scheduler）与RSDL（The Rotating Staircase Deadline Schedule）的经验 ，选取花费 CPU 执行时间最少的进程来进行调度。CFS主要由 sched_entity 内含的 vruntime所决定，不再跟踪 process 的 sleep time，并扬弃 active/expire 的概念, runqueue 里面所有的进程都平等对待，CFS 使用“虚拟运行时”virtual running time 来表示某个任务的时间量。

CFS改使用红黑树算法，将执行时间越少的工作 (即 sched_entity) 排列在红黑树的左边，时间复杂度是 O(log N)，节点 (即rb_node)的安插工作则由dequeue_entity()和enqueue_entity() 来完成。当前执行的task通过呼叫 put_prev_task 返回红黑树，下一个待执行的 task 则由 pick_next_task 来呼叫。蒙内表示, CFS 在百分之八十时间都在确实模拟处理器的处理时间。

### 争议

因为在 Linux 2.6.23 将 CFS 合并到 mainline。放弃了 RSDL，引起康恩·科里瓦斯的不满，一度宣布脱离 Linux 开发团队。数年后, Con Kolivas 卷土重来, 重新开发脑残调度器来对决 CFS, Jens Axboe 写了一个名为 Latt.c 的程序进行比对，Jens 发现 BFS 确实稍稍优于 CFS，而且 CFS 的 sleeper fairness 在某些情况下会出现调度延迟。Ingo不得不暂时关闭该特性，很快的在一周内提出新的 Gentle Fairness，彻底解决该问题。

## Reference

1. Linux 核心設計: 不只挑選任務的排程器

https://hackmd.io/@sysprog/linux-scheduler

https://www.youtube.com/watch?v=O-z0RRkW-9c

https://www.youtube.com/watch?v=2yo8Ot0FjMw

https://www.youtube.com/watch?v=-wn-ttOvQl0


2. leetcode 刷题 - 树, 红黑树, B树

https://zhuanlan.zhihu.com/p/63272157

3. Con Kolivas

https://en.wikipedia.org/wiki/Con_Kolivas

4. LKML

https://lkml.org/lkml/2004/3/24/208

5. Linux进程调度策略的发展和演变--Linux进程的管理与调度 (十六)

https://www.cnblogs.com/linhaostudy/p/9763925.html

6. 完全公平调度器 Completely Fair Scheduler

https://www.cnblogs.com/rsapaper/p/16279760.html

7. 红黑树的应用与力扣 456 号算法题 132 模式详解， go 语言解题

https://blog.csdn.net/pythonstrat/article/details/121676741

8. 2020.02.06 linux进程调度CFS 1 (红黑树)

https://zhuanlan.zhihu.com/p/105494455

9. Linux的公平调度 CFS 原理

https://www.cnblogs.com/lh03061238/p/12297214.html

10. BFS 简介，Linux 桌面的极速未来？

https://thruth.tumblr.com/post/106631835866/bfs-linux

11. Linux 核心實作 (2022): 第 7 週

https://www.youtube.com/watch?v=UpPdSqsonys

https://www.youtube.com/watch?v=yLr3qDkVEgA

https://www.youtube.com/watch?v=OeKib06vKxY


12. 你所不知道的 C 語言：記憶體管理、對齊及硬體特性

https://hackmd.io/@sysprog/c-memory?type=view

13. Linux 核心設計/實作 (Linux Kernel Internals)

http://wiki.csie.ncku.edu.tw/linux/schedule

14. Jason Note

https://www.cntofu.com/book/46/README.md




