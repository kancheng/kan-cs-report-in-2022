package main

import "fmt"


func reverse(x int) int {
    tmp := 0
    for x != 0 {
        tmp = tmp * 10 + x % 10
        x = x / 10
    }
    if tmp > 1 << 31 - 1 || tmp < -(1 << 31) {
        return 0
    }
    return 0
}

func main() {
    d := 123
    s := reverse(d)
    fmt.Println(s)
}
