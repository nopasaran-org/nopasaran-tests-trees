from tests_tree import TestsTreeNode, TestsTree

class TestsTreeTest:
    @staticmethod
    def save_tree(): # Define the root node
        root = TestsTreeNode(
        'Root',
        num_workers=2,
        inputs=[{'role': (None, False), 'client': ('client', True), 'server': ('server', True), 'ip': (None, False), 'port': (None, False), 'request-data': (None, False)}, {'role': (None, False), 'client': ('client', True), 'server': ('server', True), 'ip': (None, False), 'port': (None, False), 'response-data': (None, False)}],
        test='HTTP_CONFORMANCE'
        )

        # Initialize the tree with a repository link
        repository = "https://github.com/nopasaran-org/nopasaran-tests"
        tree = TestsTree(repository=repository)

        # Add the root node and connect the child nodes with conditions
        tree.add_root(root)

        # Save the tree as a PNG file with metadata
        png_filename = 'http_conformance.png'
        tree.save_png_with_metadata(png_filename)
        print(f"Tests tree image saved as '{png_filename}'.")

TestsTreeTest.save_tree()