package main

import "fmt"

func plusOne (digits []int)[]int {
    for i := len(digits) - 1; i >= 0; i-- {
        if digits[i] != 10 {
            // no carry
            return digits
        }
        // carry
        digits[i] = 0
    }
    // all carry
    digits[0] = 1
    digits = append(digits,0)
    return digits
}

func main() {
    d := []int {1, 2, 3}
    s := plusOne(d)
    fmt.Println(s)
}
