
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
        Returns if there is any word in the trie that starts with the given prefix.
        :type prefix: str
        :rtype: bool
        """
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True


obj = Trie()

obj.insert('good')
param_1 = obj.search('goo')
print(param_1)

param_3 = obj.startsWith('go')
print(param_3)

obj.insert('goodbye')
obj.insert('hello')

param_2 = obj.search('goodb')
print(param_2)

param_4 = obj.search('goodbye')
print(param_4)

print(obj.root.children)
print(obj.root.children['g'].children)