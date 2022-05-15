# AATCC - 算法分析和复杂性理论 - Analysis of Algorithms and Theory of Computational Complexity

> 2101212850 干皓丞

PKU 2022 個人實驗報告作業


## Details

```
bucketSort()
  create N buckets each of which can hold a range of values
  for all the buckets
    initialize each bucket with 0 values
  for all the buckets
    put elements into buckets matching the range
  for all the buckets 
    sort elements in each bucket
  gather elements from each bucket
end bucketSort
```

```
function bucket-sort(array, n) is
  buckets ← new array of n empty lists
  for i = 0 to (length(array)-1) do
    insert array[i] into buckets[msbits(array[i], k)]
  for i = 0 to n - 1 do
    next-sort(buckets[i])
  return the concatenation of buckets[0], ..., buckets[n-1]
```

### PHP

```
<?php
function insertion_sort(&$elements, $fn = 'comparison_function') {
  if (!is_array($elements) || !is_callable($fn)) return;
  for ($i = 1; $i < sizeof($elements); $i++) {
    $key = $elements[$i]; // This will be $a in comparison function.
                          // $key will be the right side element that will be
                          // compared against the left sorted elements. For
                          // min to max sort, $key should be higher than
                          // $elements[0] to $elements[$j]
    
    $j = $i - 1; // this will be in $b in comparison function
    while ( $j >= 0 && $fn($key, $elements[$j]) ) {
      $elements[$j + 1] = $elements[$j]; // shift right
      $j = $j - 1;
    }
    $elements[$j + 1] = $key;
  }
}
/**
 * Comparison function used to compare each element.
 * @param mixed $a
 * @param mixed $b
 * @returns bool True iff $a is less than $b.
 */
function comparison_function(&$a, &$b) {
  return $a < $b;
}

function bucket_sort(&$elements) {
  $n = sizeof($elements);
  $buckets = array();
  // Initialize the buckets.
  for ($i = 0; $i < $n; $i++) {
    $buckets[$i] = array();
  }
  // Put each element into matched bucket.
  foreach ($elements as $el) {
    array_push($buckets[ceil($el/10)], $el);
  }
  // Sort elements in each bucket using insertion sort.
  $j = 0;
  for ($i = 0; $i < $n; $i++) {
    // sort only non-empty bucket
    if (!empty($buckets[$i])) {
      insertion_sort($buckets[$i]);
      // Move sorted elements in the bucket into original array.
      foreach ($buckets[$i] as $el) {
        $elements[$j++] = $el;
      }
    }
  }
}
$a = array(-1,0,2,3,-2);
echo "Original Array :\n";
insertion_sort($a); // Sort the elements
echo "\nSorted Array :\n";
var_dump($a);
?>
```

```
Original Array :
array(5) {
  [0]=>
  int(-1)
  [1]=>
  int(0)
  [2]=>
  int(2)
  [3]=>
  int(3)
  [4]=>
  int(-2)
}

Sorted Array :
array(5) {
  [0]=>
  int(-2)
  [1]=>
  int(-1)
  [2]=>
  int(0)
  [3]=>
  int(2)
  [4]=>
  int(3)
}
```

### Python

```
# Bucket Sort in Python


def bucketSort(array):
    bucket = []

    # Create empty buckets
    for i in range(len(array)):
        bucket.append([])

    # Insert elements into their respective buckets
    for j in array:
        index_b = int(10 * j)
        bucket[index_b].append(j)

    # Sort the elements of each bucket
    for i in range(len(array)):
        bucket[i] = sorted(bucket[i])

    # Get the sorted elements
    k = 0
    for i in range(len(array)):
        for j in range(len(bucket[i])):
            array[k] = bucket[i][j]
            k += 1
    return array


array = [.42, .32, .33, .52, .37, .47, .51]
print("Sorted Array in descending order is")
print(bucketSort(array))
```

### Java

```
	private int indexFor(int a, int min, int step) {
		return (a - min) / step;
	}

	public void bucketSort(int[] arr) {

		int max = arr[0], min = arr[0];
		for (int a : arr) {
			if (max < a)
				max = a;
			if (min > a)
				min = a;
		}
		// 該值也可根據實際情況選擇
		int bucketNum = max / 10 - min / 10 + 1;
		List buckList = new ArrayList<List<Integer>>();
		// create bucket
		for (int i = 1; i <= bucketNum; i++) {
			buckList.add(new ArrayList<Integer>());
		}
		// push into the bucket
		for (int i = 0; i < arr.length; i++) {
			int index = indexFor(arr[i], min, 10);
			((ArrayList<Integer>) buckList.get(index)).add(arr[i]);
		}
		ArrayList<Integer> bucket = null;
		int index = 0;
		for (int i = 0; i < bucketNum; i++) {
			bucket = (ArrayList<Integer>) buckList.get(i);
			insertSort(bucket);
			for (int k : bucket) {
				arr[index++] = k;
			}
		}

	}

	// 把桶內元素插入排序
	private void insertSort(List<Integer> bucket) {
		for (int i = 1; i < bucket.size(); i++) {
			int temp = bucket.get(i);
			int j = i - 1;
			for (; j >= 0 && bucket.get(j) > temp; j--) {
				bucket.set(j + 1, bucket.get(j));
			}
			bucket.set(j + 1, temp);
		}
	}
```

```
// Bucket sort in Java

import java.util.ArrayList;
import java.util.Collections;

public class BucketSort {
  public void bucketSort(float[] arr, int n) {
    if (n <= 0)
      return;
    @SuppressWarnings("unchecked")
    ArrayList<Float>[] bucket = new ArrayList[n];

    // Create empty buckets
    for (int i = 0; i < n; i++)
      bucket[i] = new ArrayList<Float>();

    // Add elements into the buckets
    for (int i = 0; i < n; i++) {
      int bucketIndex = (int) arr[i] * n;
      bucket[bucketIndex].add(arr[i]);
    }

    // Sort the elements of each bucket
    for (int i = 0; i < n; i++) {
      Collections.sort((bucket[i]));
    }

    // Get the sorted array
    int index = 0;
    for (int i = 0; i < n; i++) {
      for (int j = 0, size = bucket[i].size(); j < size; j++) {
        arr[index++] = bucket[i].get(j);
      }
    }
  }

  // Driver code
  public static void main(String[] args) {
    BucketSort b = new BucketSort();
    float[] arr = { (float) 0.42, (float) 0.32, (float) 0.33, (float) 0.52, (float) 0.37, (float) 0.47,
        (float) 0.51 };
    b.bucketSort(arr, 7);

    for (float i : arr)
      System.out.print(i + "  ");
  }
}
```

### C

```
// Bucket sort in C

#include <stdio.h>
#include <stdlib.h>

#define NARRAY 7   // Array size
#define NBUCKET 6  // Number of buckets
#define INTERVAL 10  // Each bucket capacity

struct Node {
  int data;
  struct Node *next;
};

void BucketSort(int arr[]);
struct Node *InsertionSort(struct Node *list);
void print(int arr[]);
void printBuckets(struct Node *list);
int getBucketIndex(int value);

// Sorting function
void BucketSort(int arr[]) {
  int i, j;
  struct Node **buckets;

  // Create buckets and allocate memory size
  buckets = (struct Node **)malloc(sizeof(struct Node *) * NBUCKET);

  // Initialize empty buckets
  for (i = 0; i < NBUCKET; ++i) {
    buckets[i] = NULL;
  }

  // Fill the buckets with respective elements
  for (i = 0; i < NARRAY; ++i) {
    struct Node *current;
    int pos = getBucketIndex(arr[i]);
    current = (struct Node *)malloc(sizeof(struct Node));
    current->data = arr[i];
    current->next = buckets[pos];
    buckets[pos] = current;
  }

  // Print the buckets along with their elements
  for (i = 0; i < NBUCKET; i++) {
    printf("Bucket[%d]: ", i);
    printBuckets(buckets[i]);
    printf("\n");
  }

  // Sort the elements of each bucket
  for (i = 0; i < NBUCKET; ++i) {
    buckets[i] = InsertionSort(buckets[i]);
  }

  printf("-------------\n");
  printf("Bucktets after sorting\n");
  for (i = 0; i < NBUCKET; i++) {
    printf("Bucket[%d]: ", i);
    printBuckets(buckets[i]);
    printf("\n");
  }

  // Put sorted elements on arr
  for (j = 0, i = 0; i < NBUCKET; ++i) {
    struct Node *node;
    node = buckets[i];
    while (node) {
      arr[j++] = node->data;
      node = node->next;
    }
  }

  return;
}

// Function to sort the elements of each bucket
struct Node *InsertionSort(struct Node *list) {
  struct Node *k, *nodeList;
  if (list == 0 || list->next == 0) {
    return list;
  }

  nodeList = list;
  k = list->next;
  nodeList->next = 0;
  while (k != 0) {
    struct Node *ptr;
    if (nodeList->data > k->data) {
      struct Node *tmp;
      tmp = k;
      k = k->next;
      tmp->next = nodeList;
      nodeList = tmp;
      continue;
    }

    for (ptr = nodeList; ptr->next != 0; ptr = ptr->next) {
      if (ptr->next->data > k->data)
        break;
    }

    if (ptr->next != 0) {
      struct Node *tmp;
      tmp = k;
      k = k->next;
      tmp->next = ptr->next;
      ptr->next = tmp;
      continue;
    } else {
      ptr->next = k;
      k = k->next;
      ptr->next->next = 0;
      continue;
    }
  }
  return nodeList;
}

int getBucketIndex(int value) {
  return value / INTERVAL;
}

void print(int ar[]) {
  int i;
  for (i = 0; i < NARRAY; ++i) {
    printf("%d ", ar[i]);
  }
  printf("\n");
}

// Print buckets
void printBuckets(struct Node *list) {
  struct Node *cur = list;
  while (cur) {
    printf("%d ", cur->data);
    cur = cur->next;
  }
}

// Driver code
int main(void) {
  int array[NARRAY] = {42, 32, 33, 52, 37, 47, 51};

  printf("Initial array: ");
  print(array);
  printf("-------------\n");

  BucketSort(array);
  printf("-------------\n");
  printf("Sorted array: ");
  print(array);
  return 0;
}
```

### Cpp

```
#include<iterator>
#include<iostream>
#include<vector>
using namespace std;
const int BUCKET_NUM = 10;

struct ListNode{
	explicit ListNode(int i=0):mData(i),mNext(NULL){}
	ListNode* mNext;
	int mData;
};

ListNode* insert(ListNode* head,int val){
	ListNode dummyNode;
	ListNode *newNode = new ListNode(val);
	ListNode *pre,*curr;
	dummyNode.mNext = head;
	pre = &dummyNode;
	curr = head;
	while(NULL!=curr && curr->mData<=val){
		pre = curr;
		curr = curr->mNext;
	}
	newNode->mNext = curr;
	pre->mNext = newNode;
	return dummyNode.mNext;
}


ListNode* Merge(ListNode *head1,ListNode *head2){
	ListNode dummyNode;
	ListNode *dummy = &dummyNode;
	while(NULL!=head1 && NULL!=head2){
		if(head1->mData <= head2->mData){
			dummy->mNext = head1;
			head1 = head1->mNext;
		}else{
			dummy->mNext = head2;
			head2 = head2->mNext;
		}
		dummy = dummy->mNext;
	}
	if(NULL!=head1) dummy->mNext = head1;
	if(NULL!=head2) dummy->mNext = head2;
	
	return dummyNode.mNext;
}

void BucketSort(int n,int arr[]){
	vector<ListNode*> buckets(BUCKET_NUM,(ListNode*)(0));
	for(int i=0;i<n;++i){
		int index = arr[i]/BUCKET_NUM;
		ListNode *head = buckets.at(index);
		buckets.at(index) = insert(head,arr[i]);
	}
	ListNode *head = buckets.at(0);
	for(int i=1;i<BUCKET_NUM;++i){
		head = Merge(head,buckets.at(i));
	}
	for(int i=0;i<n;++i){
		arr[i] = head->mData;
		head = head->mNext;
	}
}
```

```
// Bucket sort in C++

#include <iomanip>
#include <iostream>
using namespace std;

#define NARRAY 7   // Array size
#define NBUCKET 6  // Number of buckets
#define INTERVAL 10  // Each bucket capacity

struct Node {
  int data;
  struct Node *next;
};

void BucketSort(int arr[]);
struct Node *InsertionSort(struct Node *list);
void print(int arr[]);
void printBuckets(struct Node *list);
int getBucketIndex(int value);

// Sorting function
void BucketSort(int arr[]) {
  int i, j;
  struct Node **buckets;

  // Create buckets and allocate memory size
  buckets = (struct Node **)malloc(sizeof(struct Node *) * NBUCKET);

  // Initialize empty buckets
  for (i = 0; i < NBUCKET; ++i) {
    buckets[i] = NULL;
  }

  // Fill the buckets with respective elements
  for (i = 0; i < NARRAY; ++i) {
    struct Node *current;
    int pos = getBucketIndex(arr[i]);
    current = (struct Node *)malloc(sizeof(struct Node));
    current->data = arr[i];
    current->next = buckets[pos];
    buckets[pos] = current;
  }

  // Print the buckets along with their elements
  for (i = 0; i < NBUCKET; i++) {
    cout << "Bucket[" << i << "] : ";
    printBuckets(buckets[i]);
    cout << endl;
  }

  // Sort the elements of each bucket
  for (i = 0; i < NBUCKET; ++i) {
    buckets[i] = InsertionSort(buckets[i]);
  }

  cout << "-------------" << endl;
  cout << "Bucktets after sorted" << endl;
  for (i = 0; i < NBUCKET; i++) {
    cout << "Bucket[" << i << "] : ";
    printBuckets(buckets[i]);
    cout << endl;
  }

  // Put sorted elements on arr
  for (j = 0, i = 0; i < NBUCKET; ++i) {
    struct Node *node;
    node = buckets[i];
    while (node) {
      arr[j++] = node->data;
      node = node->next;
    }
  }

  for (i = 0; i < NBUCKET; ++i) {
    struct Node *node;
    node = buckets[i];
    while (node) {
      struct Node *tmp;
      tmp = node;
      node = node->next;
      free(tmp);
    }
  }
  free(buckets);
  return;
}

// Function to sort the elements of each bucket
struct Node *InsertionSort(struct Node *list) {
  struct Node *k, *nodeList;
  if (list == 0 || list->next == 0) {
    return list;
  }

  nodeList = list;
  k = list->next;
  nodeList->next = 0;
  while (k != 0) {
    struct Node *ptr;
    if (nodeList->data > k->data) {
      struct Node *tmp;
      tmp = k;
      k = k->next;
      tmp->next = nodeList;
      nodeList = tmp;
      continue;
    }

    for (ptr = nodeList; ptr->next != 0; ptr = ptr->next) {
      if (ptr->next->data > k->data)
        break;
    }

    if (ptr->next != 0) {
      struct Node *tmp;
      tmp = k;
      k = k->next;
      tmp->next = ptr->next;
      ptr->next = tmp;
      continue;
    } else {
      ptr->next = k;
      k = k->next;
      ptr->next->next = 0;
      continue;
    }
  }
  return nodeList;
}

int getBucketIndex(int value) {
  return value / INTERVAL;
}

// Print buckets
void print(int ar[]) {
  int i;
  for (i = 0; i < NARRAY; ++i) {
    cout << setw(3) << ar[i];
  }
  cout << endl;
}

void printBuckets(struct Node *list) {
  struct Node *cur = list;
  while (cur) {
    cout << setw(3) << cur->data;
    cur = cur->next;
  }
}

// Driver code
int main(void) {
  int array[NARRAY] = {42, 32, 33, 52, 37, 47, 51};

  cout << "Initial array: " << endl;
  print(array);
  cout << "-------------" << endl;

  BucketSort(array);
  cout << "-------------" << endl;
  cout << "Sorted array: " << endl;
  print(array);
}
```

### JavaScript 實現算法

```
Array.prototype.bucketSort = function(num) {
  function swap(arr, i, j) {
    const temp = arr[i]
    arr[i] = arr[j]
    arr[j] = temp
  }
  const max = Math.max(...this)
  const min = Math.min(...this)
  const buckets = []
  const bucketsSize = Math.floor((max - min) / num) + 1
  for(let i = 0; i < this.length; i++) {
    const index = ~~(this[i] / bucketsSize)
    !buckets[index] && (buckets[index] = [])
    buckets[index].push(this[i])
    let l = buckets[index].length
    while(l > 0) {
      buckets[index][l] < buckets[index][l - 1] && swap(buckets[index], l, l - 1)
      l--
    }
  }
  let wrapBuckets = []
  for(let i = 0; i < buckets.length; i++) {
    buckets[i] && (wrapBuckets = wrapBuckets.concat(buckets[i]))
  }
  return wrapBuckets
}
const arr = [11, 9, 6, 8, 1, 3, 5, 1, 1, 0, 100]
console.log(arr.bucketSort(10))
```

### JavaScript 實現遞歸算法

```
Array.prototype.bucketSort = function()
{
  var start = 0;
  var size = this.length;
  var min = this[0];
  var max = this[0]; 
  for (var i = 1; i < size; i++){
    if (this[i] < min){min = this[i];}
    else{ if(this[i] > max){max = this[i];} }
  }
  if (min != max){
    var bucket = new Array(size);
    for (var i = 0; i < size; i++){bucket[i] = new Array();}
    var interpolation = 0;
    for (var i = 0; i < size; i++){
      interpolation = Math.floor(((this[i] - min) / (max - min)) * (size - 1));
      bucket[interpolation].push(this[i]);
    }
    for (var i = 0; i < size; i++){
      if (bucket[i].length > 1){bucket[i].bucketSort();}//遞歸
      for(var j = 0; j < bucket[i].length; j++){this[start++] = bucket[i][j];}
    }
  }
};
```

## Reference

1. https://en.wikipedia.org/wiki/Bucket_sort

2. https://zh.m.wikipedia.org/zh-hant/%E6%A1%B6%E6%8E%92%E5%BA%8F

3. https://www.programiz.com/dsa/bucket-sort

4. https://www.w3resource.com/php-exercises/searching-and-sorting-algorithm/searching-and-sorting-algorithm-exercise-10.php











