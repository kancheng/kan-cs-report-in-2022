class Solution(object):
    def longestWord(self, words):

        valid = set([""])

        for word in sorted(words, key=len):
            if word[:-1] in valid:
                valid.add(word)

        return max(sorted(valid), key=len)


if __name__ == '__main__':
    words = ["a", "ba", "banana", "app", "appl", "ap", "apply", "apple", "appla"]
    print(Solution().longestWord(words))

