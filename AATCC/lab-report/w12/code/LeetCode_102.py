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


